import os
import pybullet as p
import numpy as np
from my_envs.envs.util import enforce_joint_limits,get_pos
class Motion:
    def __init__(self,pid,np_random):
        self.id = pid
        self.robot_right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50]  # 从urdf中得到的关节id
        self.robot_left_arm_joint_indices = [64, 65, 66, 68, 69, 71, 72]
        self.np_random=np_random

    def reset_robot_joints(self,robot):
        # Reset all robot joints
        for rj in range(p.getNumJoints(robot, physicsClientId=self.id)):
            p.resetJointState(robot, jointIndex=rj, targetValue=0, targetVelocity=0, physicsClientId=self.id)
        # Position end effectors whith dual arm robots
        for i, j in enumerate(self.robot_left_arm_joint_indices):
            p.resetJointState(robot, jointIndex=j, targetValue=[1.75, 1.25, 1.5, -0.5, 1, 0, 1][i], targetVelocity=0, physicsClientId=self.id)
        for i, j in enumerate(self.robot_right_arm_joint_indices):
            p.resetJointState(robot, jointIndex=j, targetValue=[-1.75, 1.25, -1.5, -0.5, -1, 0, -1][i], targetVelocity=0, physicsClientId=self.id)


    def reset_human_joints(self, human, controllable_joints):
        # 20 neck 21-23 head x,y,z
        joints_positions = [(6, np.deg2rad(-90)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
        joints_positions += [(21, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (22, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (23, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
        # Set starting joint positions
        human_joint_positions = get_pos(human,list(range(p.getNumJoints(human, physicsClientId=self.id))),self.id)
        for j in range(p.getNumJoints(human, physicsClientId=self.id)):
            set_position = None
            for j_index, j_angle in joints_positions:
                if j == j_index:
                    p.resetJointState(human, jointIndex=j, targetValue=j_angle, targetVelocity=0,
                                      physicsClientId=self.id)
                    set_position = j_angle
                    break
            if  j not in controllable_joints:
                p.changeDynamics(human, j, mass=0, physicsClientId=self.id)  # 改变特性
                # Set velocities to 0
                p.resetJointState(human, jointIndex=j,
                                  targetValue=human_joint_positions[j] if set_position is None else set_position,
                                  targetVelocity=0, physicsClientId=self.id)

        for j in range(p.getNumJoints(human, physicsClientId=self.id)):
            p.setJointMotorControl2(human, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=self.id)
        enforce_joint_limits(human,self.id)
        p.resetBasePositionAndOrientation(human, [0, 0.03, 0.89 ], [0, 0, 0, 1],
                                          physicsClientId=self.id)


    def reset_gripper(self, robot, position=0):
        indices=  [57, 58, 59, 60]
        positions = [position]*len(indices)
        for i, j in enumerate(indices):
            p.resetJointState(robot, jointIndex=j, targetValue=positions[i], targetVelocity=0, physicsClientId=self.id)
        p.setJointMotorControlArray(robot, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=positions, positionGains=np.array([0.05]*len(indices)), forces=[500]*len(indices), physicsClientId=self.id)

