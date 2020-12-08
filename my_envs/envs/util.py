import numpy as np
import pybullet as p

class Util:
    def __init__(self, pid, np_random):
        self.id = pid
        self.ik_lower_limits = {}
        self.ik_upper_limits = {}
        self.ik_joint_ranges = {}
        self.ik_rest_poses = {}
        self.np_random = np_random

    def ik_jlwki(self, body, target_joint, target_pos, target_orient, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, ik_indices=range(29, 29+7), max_iterations=100, success_threshold=0.03, half_range=False, step_sim=False, check_env_collisions=False):
        target_joint_positions = self.ik(body, target_joint, target_pos, target_orient, ik_indices=ik_indices, max_iterations=max_iterations, half_range=half_range)
        setup_robot_joints(self.id,body, robot_arm_joint_indices, robot_lower_limits, robot_upper_limits, default_positions=np.array(target_joint_positions))
        if step_sim:
            for _ in range(5):
                p.stepSimulation(physicsClientId=self.id)
            if len(p.getContactPoints(bodyA=body, bodyB=body, physicsClientId=self.id)) > 0:#接触点
                # The robot's arm is in contact with itself.
                return False, np.array(target_joint_positions)
        if check_env_collisions:
            for _ in range(25):
                p.stepSimulation(physicsClientId=self.id)
        gripper_pos, gripper_orient = p.getLinkState(body, target_joint, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        if np.linalg.norm(target_pos - np.array(gripper_pos)) < success_threshold and (target_orient is None or np.linalg.norm(target_orient - np.array(gripper_orient)) < success_threshold or np.isclose(np.linalg.norm(target_orient - np.array(gripper_orient)), 2, atol=success_threshold)):
            return True, np.array(target_joint_positions)
        return False, np.array(target_joint_positions)

    def ik(self, body, target_joint, target_pos, target_orient, ik_indices=range(29, 29+7), max_iterations=1000, half_range=False):
        key = '%d_%d' % (body, target_joint)
        if key not in self.ik_lower_limits:
            self.ik_lower_limits[key] = []
            self.ik_upper_limits[key] = []
            self.ik_joint_ranges[key] = []
            self.ik_rest_poses[key] = []
            j_names = []
            for j in range(p.getNumJoints(body, physicsClientId=self.id)):
                if p.getJointInfo(body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:#非固定关节
                    joint_info = p.getJointInfo(body, j, physicsClientId=self.id)
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    if lower_limit == 0 and upper_limit == -1:
                        lower_limit = -2*np.pi
                        upper_limit = 2*np.pi
                    self.ik_lower_limits[key].append(lower_limit)
                    self.ik_upper_limits[key].append(upper_limit)
                    if not half_range:
                        self.ik_joint_ranges[key].append(upper_limit - lower_limit)
                    else:
                        self.ik_joint_ranges[key].append((upper_limit - lower_limit)/2.0)
                    j_names.append([len(j_names)] + list(joint_info[:2]))
        self.ik_rest_poses[key] = self.np_random.uniform(self.ik_lower_limits[key], self.ik_upper_limits[key]).tolist()#可行的reset范围
        if target_orient is not None:
            ik_joint_poses = np.array(p.calculateInverseKinematics(body, target_joint, targetPosition=target_pos, targetOrientation=target_orient, lowerLimits=self.ik_lower_limits[key], upperLimits=self.ik_upper_limits[key], jointRanges=self.ik_joint_ranges[key], restPoses=self.ik_rest_poses[key], maxNumIterations=max_iterations, physicsClientId=self.id))
        else:
            ik_joint_poses = np.array(p.calculateInverseKinematics(body, target_joint, targetPosition=target_pos, lowerLimits=self.ik_lower_limits[key], upperLimits=self.ik_upper_limits[key], jointRanges=self.ik_joint_ranges[key], restPoses=self.ik_rest_poses[key], maxNumIterations=max_iterations, physicsClientId=self.id))
        target_joint_positions = ik_joint_poses[ik_indices]
        return target_joint_positions


def get_pos(object, indice,id):
    states = p.getJointStates(object, jointIndices=indice,
                              physicsClientId=id)
    positions = np.array([x[0] for x in states])
    return positions

def get_motor_joint_states(object, id):
    num_joints = p.getNumJoints(object, physicsClientId=id)
    joint_states = p.getJointStates(object, range(num_joints), physicsClientId=id)
    joint_infos = [p.getJointInfo(object, i, physicsClientId=id) for i in range(num_joints)]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


def enforce_joint_limits(body,id):
    # Enforce joint limits
    joint_positions = get_pos(body, list(range(p.getNumJoints(body, physicsClientId=id))),id)
    lower_limits = []
    upper_limits = []
    for j in range(p.getNumJoints(body, physicsClientId=id)):
        joint_info = p.getJointInfo(body, j, physicsClientId=id)
        joint_pos = joint_positions[j]
        lower_limit = joint_info[8]  # Positional lower limit
        upper_limit = joint_info[9]
        if lower_limit == 0 and upper_limit == -1:
            lower_limit = -1e10
            upper_limit = 1e10
        lower_limits.append(lower_limit)
        upper_limits.append(upper_limit)
        if joint_pos < lower_limit:
            p.resetJointState(body, jointIndex=j, targetValue=lower_limit, targetVelocity=0,
                              physicsClientId=id)
        elif joint_pos > upper_limit:
            p.resetJointState(body, jointIndex=j, targetValue=upper_limit, targetVelocity=0,
                              physicsClientId=id)
    lower_limits = np.array(lower_limits)
    upper_limits = np.array(upper_limits)
    return lower_limits, upper_limits

def setup_robot_joints(id, robot, robot_joint_indices, lower_limits, upper_limits, default_positions=[1, 1, 0, -1.75, 0, -1.1, -0.5]):
    default_positions[default_positions < lower_limits] = lower_limits[default_positions < lower_limits]
    default_positions[default_positions > upper_limits] = upper_limits[default_positions > upper_limits]
    for i, j in enumerate(robot_joint_indices):
        p.resetJointState(robot, jointIndex=j, targetValue=default_positions[i], targetVelocity=0, physicsClientId=id)

