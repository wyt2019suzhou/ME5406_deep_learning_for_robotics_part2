import os, time, datetime, configparser
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
from screeninfo import get_monitors
from my_envs.envs.objectmotion import Motion
from my_envs.envs.createobject import Create
from my_envs.envs.util import get_motor_joint_states,get_pos,Util,setup_robot_joints
import cv2

class BaseEnv(gym.Env):
    def __init__(self, human_control=False, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=4, obs_robot_len=24, obs_human_len=21):
        # Start the bullet physics server
        self.id = p.connect(p.DIRECT)
        self.gui=False
        self.human_control = human_control
        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len
        self.action_human_len = action_human_len
        self.obs_human_len = obs_human_len

        self.action_space = spaces.Box(low=np.array([-1.0]*(self.action_robot_len+self.action_human_len)), high=np.array([1.0]*(self.action_robot_len+self.action_human_len)))
        self.observation_space = spaces.Box(low=np.array([-1.0]*(self.obs_robot_len+self.obs_human_len)), high=np.array([1.0]*(self.obs_robot_len+self.obs_human_len)))

        self.configp = configparser.ConfigParser()
        self.configp.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))

        # Execute actions at 10 Hz by default. A new action every 0.1 seconds
        self.frame_skip = frame_skip
        self.time_step = time_step

        self.last_sim_time = None
        self.task_success =0
        self.seed(1001)

        self.builder = Create(self.id, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.motion = Motion(self.id,np_random=self.np_random)
        self.util = Util(self.id, self.np_random)


        self.record_video = False
        self.video_writer = None

        self.width = get_monitors()[0].width
        self.height = get_monitors()[0].height

        self.right_arm_previous_valid_pose = None
        self.left_arm_previous_valid_pose = None
        self.human_joint_lower_limits = None
        self.human_joint_upper_limits = None
        self.human_controllable_joint_indices = [20, 21, 22, 23]
        self.robot_right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50]  # 从urdf中得到的关节id
        self.robot_left_arm_joint_indices = [64, 65, 66, 68, 69, 71, 72]
        self.mouth_pos=[0, -0.11, 0.03]
        self.time_penalty=self.config('time_penalty','feeding')
        self.dis_weight=self.config('dis_weight','feeding')
        self.food_weight=self.config('food_weight','feeding')
        self.action_weight=self.config('action_weight','feeding')
        self.spoon_weight=self.config('spoon_weight','feeding')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def config(self, tag, section=None):
        return float(self.configp[section][tag])

    def take_step(self, action, gains=0.05, forces=1, human_gains=0.1, human_forces=1, step_sim=True):
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        action=np.nan_to_num(action)
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        action *= 0.05
        action_robot = action

        if self.human_control:
            action_robot = action[:self.action_robot_len]
            action_human = action[self.action_robot_len:]
            human_joint_positions = get_pos( self.human,self.human_controllable_joint_indices,self.id)

        robot_joint_positions = get_pos(self.robot,self.robot_right_arm_joint_indices,self.id)

        for _ in range(self.frame_skip):
            action_robot[robot_joint_positions + action_robot < self.robot_lower_limits] = 0#超出限制则不动
            action_robot[robot_joint_positions + action_robot > self.robot_upper_limits] = 0
            robot_joint_positions += action_robot
            if self.human_control :
                action_human[human_joint_positions + action_human < self.human_lower_limits] = 0
                action_human[human_joint_positions + action_human > self.human_upper_limits] = 0
                human_joint_positions += action_human

        p.setJointMotorControlArray(self.robot, jointIndices=self.robot_right_arm_joint_indices, controlMode=p.POSITION_CONTROL, targetPositions=robot_joint_positions, positionGains=np.array([gains]*self.action_robot_len), forces=[forces]*self.action_robot_len, physicsClientId=self.id)
        if self.human_control:
            p.setJointMotorControlArray(self.human, jointIndices=self.human_controllable_joint_indices, controlMode=p.POSITION_CONTROL, targetPositions=human_joint_positions, positionGains=np.array([human_gains]*self.action_human_len), forces=[human_forces] *self.action_human_len, physicsClientId=self.id)

        if step_sim:
            # Update robot position
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                self.enforce_hard_human_joint_limits()
                self.update_targets()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()


    def joint_limited_weighting(self, q, lower_limits, upper_limits):
        phi = 0.5
        lam = 0.05
        weights = []
        for qi, l, u in zip(q, lower_limits, upper_limits):
            qr = 0.5*(u - l)
            weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
            if weights[-1] < 0.001:
                weights[-1] = 0.001
        # Joint-limited-weighting
        joint_limit_weight = np.diag(weights)
        return joint_limit_weight


    def position_robot_toc(self, robot, joints, start_pos_orient, target_pos_orients, joint_indices, lower_limits, upper_limits, ik_indices, pos_offset=np.zeros(3), base_euler_orient=np.zeros(3), max_ik_iterations=500, attempts=100, ik_random_restarts=1, step_sim=False, check_env_collisions=False, right_side=True, random_rotation=30, random_position=0.5, human_joint_indices=None, human_joint_positions=None):
        # Continually randomize the robot base position and orientation
        # Select best base pose according to number of goals reached and manipulability
        joints = [joints]
        start_pos_orient = [start_pos_orient]
        target_pos_orients = [target_pos_orients]
        joint_indices = [joint_indices]
        lower_limits = [lower_limits]
        upper_limits = [upper_limits]
        ik_indices = [ik_indices]#可活动关节
        a = 6 # Order of the robot space. 6D (3D position, 3D orientation)
        best_position = None
        best_orientation = None
        best_num_goals_reached = None
        best_manipulability = None
        best_start_joint_poses = [None]*len(joints)
        start_fails = 0
        iteration = 0
        best_pose_count = 0
        while iteration < attempts or best_position is None:
            iteration += 1
            random_pos = np.array([self.np_random.uniform(-random_position if right_side else 0, 0 if right_side else random_position), self.np_random.uniform(-random_position, random_position), 0])
            random_orientation = p.getQuaternionFromEuler([base_euler_orient[0], base_euler_orient[1], base_euler_orient[2] + np.deg2rad(self.np_random.uniform(-random_rotation, random_rotation))], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(robot, np.array([-0.85, -0.4, 0]) + pos_offset + random_pos, random_orientation, physicsClientId=self.id)#移动机器人到人身边
            # Check if the robot can reach all target locations from this base pose
            num_goals_reached = 0
            manipulability = 0.0
            start_joint_poses = [None]*len(joints)
            for i, joint in enumerate(joints):#检查每一个是否能从开始位置到达目标位置
                for j, (target_pos, target_orient) in enumerate(start_pos_orient[i] + target_pos_orients[i]):
                    best_jlwki = None
                    goal_success = False
                    orient = target_orient
                    for k in range(ik_random_restarts):
                        # Reset human joints in case they got perturbed by previous iterations
                        if human_joint_positions is not None:
                            for h, pos in zip(human_joint_indices, human_joint_positions):
                                p.resetJointState(self.human, jointIndex=h, targetValue=pos, targetVelocity=0, physicsClientId=self.id)
                        # Reset all robot joints
                        self.motion.reset_robot_joints(robot)
                        # Find IK solution
                        success, joint_positions_q_star = self.util.ik_jlwki(robot, joint, target_pos, orient,joint_indices[i], lower_limits[i], upper_limits[i], ik_indices=ik_indices[i], max_iterations=max_ik_iterations, success_threshold=0.03, half_range=False, step_sim=step_sim, check_env_collisions=check_env_collisions)
                        if success:
                            goal_success = True
                        else:
                            goal_success = False#break ik_random_restarts
                            break
                        joint_positions, _, _ = get_motor_joint_states(robot,self.id)
                        joint_velocities = [0.0] * len(joint_positions)
                        joint_accelerations = [0.0] * len(joint_positions)
                        center_of_mass = p.getLinkState(robot, joint, computeLinkVelocity=True, computeForwardKinematics=True, physicsClientId=self.id)[2]
                        J_linear, J_angular = p.calculateJacobian(robot, joint, localPosition=center_of_mass, objPositions=joint_positions, objVelocities=joint_velocities, objAccelerations=joint_accelerations, physicsClientId=self.id)
                        J_linear = np.array(J_linear)[:, ik_indices[i]]
                        J_angular = np.array(J_angular)[:, ik_indices[i]]
                        J = np.concatenate([J_linear, J_angular], axis=0)
                        # Joint-limited-weighting
                        joint_limit_weight = self.joint_limited_weighting(joint_positions_q_star, lower_limits[i], upper_limits[i])
                        # Joint-limited-weighted kinematic isotropy (JLWKI)
                        det = np.linalg.det(np.matmul(np.matmul(J, joint_limit_weight), J.T))
                        if det < 0:
                            det = 0
                        jlwki = np.power(det, 1.0/a) / (np.trace(np.matmul(np.matmul(J, joint_limit_weight), J.T))/a)
                        if best_jlwki is None or jlwki > best_jlwki:
                            best_jlwki = jlwki
                    if goal_success:
                        num_goals_reached += 1
                        manipulability += best_jlwki
                        if j == 0:
                            start_joint_poses[i] = joint_positions_q_star
                    if j < len(start_pos_orient[i]) and not goal_success:
                        # Not able to find an IK solution to a start goal. We cannot use this base pose
                        start_fails += 1
                        num_goals_reached = -1
                        manipulability = None
                        break
                if num_goals_reached == -1:
                    break

            if num_goals_reached == 4:#在此之前就运动
                best_pose_count += 1
            if num_goals_reached > 0:
                if best_position is None or num_goals_reached > best_num_goals_reached or (num_goals_reached == best_num_goals_reached and manipulability > best_manipulability):
                    best_position = random_pos
                    best_orientation = random_orientation
                    best_num_goals_reached = num_goals_reached
                    best_manipulability = manipulability
                    best_start_joint_poses = start_joint_poses

        p.resetBasePositionAndOrientation(robot, np.array([-0.85, -0.4, 0]) + pos_offset + best_position, best_orientation, physicsClientId=self.id)
        for i, joint in enumerate(joints):
            setup_robot_joints(self.id,robot, joint_indices[i], lower_limits[i], upper_limits[i],  default_positions=np.array(best_start_joint_poses[i]))
        # Reset human joints in case they got perturbed by previous iterations
        if human_joint_positions is not None:
            for h, pos in zip(human_joint_indices, human_joint_positions):
                p.resetJointState(self.human, jointIndex=h, targetValue=pos, targetVelocity=0, physicsClientId=self.id)
        return best_position, best_orientation, best_start_joint_poses#robot的best，已经重置好

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()


    def setup_record_video(self):
        if self.record_video :
            if self.video_writer is not None:
                self.video_writer.release()
            now = datetime.datetime.now()
            date = now.strftime('%Y-%m-%d_%H-%M-%S')
            self.video_writer = cv2.VideoWriter('%s_%s.avi' % (date), cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.width, self.height))

    def record_video_frame(self):
        if self.record_video :
            frame = np.reshape(p.getCameraImage(width=self.width, height=self.height, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.id)[2], (self.height, self.width, 4))[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def update_targets(self):
        pass

    def render(self, mode='human'):
        self.id = p.connect(p.GUI)
        self.gui = True
        self.builder = Create(self.id, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.motion = Motion(self.id, np_random=self.np_random)
        self.util = Util(self.id, self.np_random)

    def play_show(self):
        self.id = p.connect(p.GUI)
        self.gui = True
        self.builder = Create(self.id, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.motion = Motion(self.id, np_random=self.np_random)
        self.util = Util(self.id, self.np_random)

    def close_render(self):
        self.id = p.connect(p.DIRECT)
        self.gui = False
        self.builder = Create(self.id, time_step=self.time_step, np_random=self.np_random, config=self.config)
        self.motion = Motion(self.id, np_random=self.np_random)
        self.util = Util(self.id, self.np_random)


    def enforce_hard_human_joint_limits(self):
        joint_positions= get_pos(self.human,self.human_controllable_joint_indices,self.id)
        if self.human_joint_lower_limits is None:
            self.human_joint_lower_limits = []
            self.human_joint_upper_limits = []
            for i, j in enumerate(self.human_controllable_joint_indices):
                joint_info = p.getJointInfo(self.human, j, physicsClientId=self.id)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.human_joint_lower_limits.append(lower_limit)
                self.human_joint_upper_limits.append(upper_limit)
        for i, j in enumerate(self.human_controllable_joint_indices):
            if joint_positions[i] < self.human_joint_lower_limits[i]:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.human_joint_lower_limits[i], targetVelocity=0, physicsClientId=self.id)
            elif joint_positions[i] > self.human_joint_upper_limits[i]:
                p.resetJointState(self.human, jointIndex=j, targetValue=self.human_joint_upper_limits[i], targetVelocity=0, physicsClientId=self.id)