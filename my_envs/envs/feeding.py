import os
from gym import spaces
import numpy as np
import pybullet as p
from my_envs.envs.util import get_pos
from my_envs.envs.env import BaseEnv

class FeedingEnv(BaseEnv):
    def __init__(self, human_control=False):
        super(FeedingEnv, self).__init__( human_control=human_control, frame_skip=10, time_step=0.01, action_robot_len=7, action_human_len=(4 if human_control else 0), obs_robot_len=24, obs_human_len=(21 if human_control else 0))

    def step(self, action):
        self.take_step(action, gains=self.config('robot_gains','feeding'), forces=self.config('robot_forces','feeding'), human_gains=0.0005)
        reward_food ,currentDistance= self.get_food_rewards()
        reward_action=-np.sum(np.square(action))
        reward_dis = -(currentDistance - self.previousDistance)
        self.previousDistance = currentDistance
        reward=self.food_weight*reward_food+self.dis_weight*reward_dis-self.time_penalty+self.action_weight*reward_action
        obs = self._get_obs()
        info = {'success_time': self.task_success,'fall_time': self.fall_time}
        if len(self.foods) ==0:
            done=True
        else:
            done = False

        return obs, reward, done, info


    def get_food_rewards(self):
        food_reward = 0
        foods_to_remove = []
        sum_dis = 0
        for f in self.foods:
            food_pos, food_orient = p.getBasePositionAndOrientation(f, physicsClientId=self.id)
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            sum_dis+=distance_to_mouth
            if distance_to_mouth < 0.03:
                # Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                foods_to_remove.append(f)
                p.resetBasePositionAndOrientation(f, self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1], physicsClientId=self.id)#重置foodposition,远去
                continue
            elif food_pos[-1] < 0.5 or len(p.getContactPoints(bodyA=f, bodyB=self.table, physicsClientId=self.id)) > 0 or len(p.getContactPoints(bodyA=f, bodyB=self.bowl, physicsClientId=self.id)) > 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                self.fall_time+=1
                foods_to_remove.append(f)
        mean_dis=sum_dis/len(self.foods)
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        return food_reward,mean_dis

    def _get_obs(self):
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        head_pos,head_orient=p.getLinkState(self.human,23,computeForwardKinematics=True, physicsClientId=self.id)[:2]
        robot_right_joint_positions =get_pos(self.robot, self.robot_right_arm_joint_indices,self.id)
        if self.human_control:
            human_joint_positions =get_pos(self.human,self.human_controllable_joint_indices,self.id)
        robot_obs = np.concatenate(
            [spoon_pos , spoon_pos - self.target_pos, spoon_orient, robot_right_joint_positions,
             head_pos, head_orient]).ravel()  # 3+4+7+3+4=24
        if self.human_control:
            human_obs = np.concatenate([spoon_pos, spoon_pos-self.target_pos,spoon_orient, human_joint_positions,head_pos, head_orient]).ravel()#21
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()#45

    def reset(self):
        self.last_sim_time = None
        self.task_success=0
        self.fall_time=0
        self.human,self.robot, self.wheelchair,self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits= self.builder.create_base()
        self.motion.reset_robot_joints(self.robot)
        self.motion.reset_human_joints(self.human, self.human_controllable_joint_indices if self.human_control else [] )
        self.target_human_joint_positions =get_pos(self.human, self.human_controllable_joint_indices,self.id)
        self.table= self.builder.create_table()
        self.bowl = self.builder.create_bowl()
        self.target,self.target_pos=self.builder.create_target(self.human)

        bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array(
            [self.np_random.uniform(-0.05, 0.05), self.np_random.uniform(-0.05, 0.05), 0])
        pose_pos = np.array(bowl_pos) + np.array([0, -0.1, 0.4]) + self.np_random.uniform(-0.05, 0.05, size=3)
        pos_orient = p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
        self.position_robot_toc(self.robot, 54, [(pose_pos, pos_orient), (self.target_pos, None)], [(self.target_pos, pos_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(15, 15+7), pos_offset=np.array([0.1, 0.2, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
        self.motion.reset_gripper(self.robot, position=0.03)
        self.spoon = self.builder.create_spoon(self.robot, mesh_scale=[0.08] * 3, pos_offset=[0, -0.03, -0.11], orient_offset=p.getQuaternionFromEuler([-0.2, 0, 0], physicsClientId=self.id))
        p.resetBasePositionAndOrientation(self.bowl, bowl_pos,
                                          p.getQuaternionFromEuler([np.pi / 2.0, 0, 0], physicsClientId=self.id),
                                          physicsClientId=self.id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)

        self.foods=self.builder.create_food(self.spoon)

        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)#numSubStep streat performance more acurate,
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)#加入shadow等试试

        # Drop food in the spoon
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
        sum_dis=0
        for f in self.foods:
            food_pos, food_orient = p.getBasePositionAndOrientation(f, physicsClientId=self.id)
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            sum_dis += distance_to_mouth
        self.previousDistance=sum_dis/len(self.foods)

        return self._get_obs()

    def update_targets(self):
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

    def success_time(self):
        return self.task_success

    def fall_times(self):
        return self.fall_time
