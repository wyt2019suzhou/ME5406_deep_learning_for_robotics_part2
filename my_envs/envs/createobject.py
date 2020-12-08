import os
import numpy as np
import pybullet as p
from my_envs.envs.util import get_pos,enforce_joint_limits


# -- Joint Legend --

# 0-2 right_shoulder x,y,z
# 3-5 right_shoulder_socket x,y,z
# 6 right_elbow x
# 7 right_forearm_roll z
# 8-9 right_hand x,y
# 10-12 left_shoulder x,y,z
# 13-15 left_shoulder_socket x,y,z
# 16 left_elbow x
# 17 left_forearm_roll z
# 18-19 left_hand x,y
# 20 neck x
# 21-23 head x,y,z
# 25-27 waist x,y,z
# 28-30 right_hip x,y,z
# 31 right_knee x
# 32-34 right_ankle x,y,z
# 35-37 left_hip x,y,z
# 38 left_knee x
# 39-41 left_ankle x,y,z

# -- Limb (link) Legend --

# 2 right_shoulder  up
# 5 right_upperarm   up
# 7 right_forearm
# 9 right_hand
# 12 left_shoulder  up
# 15 left_upperarm
# 17 left_forearm
# 19 left_hand
# 20 neck
# 23 head
# 24 waist   up
# 27 hips    down
# 30 right_thigh down
# 31 right_shin
# 34 right_foot
# 37 left_thigh
# 38 left_shin
# 41 left_foot
# -1 belly up

class Create:
    def __init__(self, pid, np_random=None, time_step=0.02, config=None):
        self.id = pid
        self.np_random = np_random
        self.config = config
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.human_limit_scale = 1.0
        self.time_step = time_step
        self.human_controllable_joint_indices = [20, 21, 22, 23]
        self.robot_right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50]  # 从urdf中得到的关节id
        self.robot_left_arm_joint_indices = [64, 65, 66, 68, 69, 71, 72]

    def create_base(self):
        p.resetSimulation(physicsClientId=self.id)
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45,
                                     cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)  # 在创建过程中关闭可视化
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)  # 在创建过程中关闭可视化

        self.createplane()
        wheelchair = self.createwheelchair()

        human = self.creathuman(self.human_limit_scale)
        human_lower_limits, human_upper_limits =enforce_joint_limits(human,self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        human_lower_limits = human_lower_limits[self.human_controllable_joint_indices]
        human_upper_limits = human_upper_limits[self.human_controllable_joint_indices]

        robot = self.createrobot()
        robot_lower_limits, robot_upper_limits = enforce_joint_limits(robot,self.id)
        robot_lower_limits = robot_lower_limits[self.robot_right_arm_joint_indices]
        robot_upper_limits = robot_upper_limits[self.robot_right_arm_joint_indices]

        return human, robot, wheelchair, robot_lower_limits,robot_upper_limits, human_lower_limits, human_upper_limits

    def creathuman(self, limit_scale=1.0):
        joint_c, joint_v = -1, -1  # collision，visual
        m = self.config('mass', 'human')
        rs = self.config('radius_scale', 'human')
        hs = self.config('height_scale', 'human')
        chest_c, chest_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.127 * rs, length=0.056,
                                            orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
                                                                                 physicsClientId=self.id))
        right_shoulders_c, right_shoulders_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.106 * rs,
                                                                length=0.253 / 8,
                                                                position_offset=[-0.253 / 2.5 + 0.253 / 16, 0, 0],
                                                                orientation=p.getQuaternionFromEuler(
                                                                    [0, np.pi / 2.0, 0], physicsClientId=self.id))
        left_shoulders_c, left_shoulders_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.106 * rs,
                                                              length=0.253 / 8,
                                                              position_offset=[0.253 / 2.5 - 0.253 / 16, 0, 0],
                                                              orientation=p.getQuaternionFromEuler(
                                                                  [0, np.pi / 2.0, 0], physicsClientId=self.id))
        neck_c, neck_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.06 * rs, length=0.124 * hs,
                                          position_offset=[0, 0, (0.2565 - 0.1415 - 0.025) * hs])
        upperarm_c, upperarm_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.043 * rs, length=0.279 * hs,
                                                  position_offset=[0, 0, -0.279 / 2.0 * hs])
        forearm_c, forearm_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.033 * rs, length=0.257 * hs,
                                                position_offset=[0, 0, -0.257 / 2.0 * hs])
        hand_c, hand_v = self.create_body(shape=p.GEOM_SPHERE, radius=0.043 * rs, length=0,
                                          position_offset=[0, 0, -0.043 * rs])
        waist_c, waist_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.1205 * rs, length=0.049,
                                            orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
                                                                                 physicsClientId=self.id))
        hips_c, hips_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.1335 * rs, length=0.094,
                                          position_offset=[0, 0, -0.08125 * hs],
                                          orientation=p.getQuaternionFromEuler([0, np.pi / 2.0, 0],
                                                                               physicsClientId=self.id))
        thigh_c, thigh_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.08 * rs, length=0.424 * hs,
                                            position_offset=[0, 0, -0.424 / 2.0 * hs])
        shin_c, shin_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.05 * rs, length=0.403 * hs,
                                          position_offset=[0, 0, -0.403 / 2.0 * hs])
        foot_c, foot_v = self.create_body(shape=p.GEOM_CAPSULE, radius=0.05 * rs, length=0.215 * hs,
                                          position_offset=[0, -0.1, -0.025 * rs],
                                          orientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                               physicsClientId=self.id))
        elbow_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=(0.043 + 0.033) / 2 * rs, length=0,
                                      rgbaColor=[0.8, 0.6, 0.4, 1], visualFramePosition=[0, 0.01, 0],
                                      physicsClientId=self.id)

        self.hand_radius, self.elbow_radius, self.shoulder_radius = 0.043 * rs, 0.043 * rs, 0.043 * rs
        head_scale = [0.89] * 3
        head_pos = [0.09, 0.08, -0.07 + 0.01]
        head_c = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                        fileName=os.path.join(self.directory, 'head_female_male',
                                                              'BaseHeadMeshes_v5_male_cropped_reduced_compressed_vhacd.obj'),
                                        collisionFramePosition=head_pos,
                                        collisionFrameOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                                           physicsClientId=self.id),
                                        meshScale=head_scale, physicsClientId=self.id)
        head_v = p.createVisualShape(shapeType=p.GEOM_MESH,
                                     fileName=os.path.join(self.directory, 'head_female_male',
                                                           'BaseHeadMeshes_v5_male_cropped_reduced_compressed.obj'),
                                     rgbaColor=[0.8, 0.6, 0.4, 1], specularColor=[0.1, 0.1, 0.1],
                                     visualFramePosition=head_pos,
                                     visualFrameOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                                     physicsClientId=self.id),
                                     meshScale=head_scale, physicsClientId=self.id)
        # init position and orentation
        joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]  # position orientation
        chest_p = [0, 0, 1.2455 * hs]
        shoulders_p = [0, 0, 0.1415 / 2 * hs]
        neck_p = [0, 0, 0.1515 * hs]
        head_p = [0, 0, (0.399 - 0.1415 - 0.1205) * hs]
        right_upperarm_p = [-0.106 * rs - 0.073, 0, 0]
        left_upperarm_p = [0.106 * rs + 0.073, 0, 0]
        forearm_p = [0, 0, -0.279 * hs]
        hand_p = [0, 0, -(0.033 * rs + 0.257 * hs)]
        waist_p = [0, 0, -0.156 * hs]
        hips_p = [0, 0, -0.08125 * hs]
        right_thigh_p = [-0.08 * rs - 0.009, 0, -0.08125 * hs]
        left_thigh_p = [0.08 * rs + 0.009, 0, -0.08125 * hs]
        shin_p = [0, 0, -0.424 * hs]
        foot_p = [0, 0, -0.403 * hs - 0.025]

        # createMultiBody
        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []
        linkLowerLimits = []
        linkUpperLimits = []

        # NOTE: Shoulders, neck, and head
        linkMasses.extend(m * np.array([0, 0, 0.05, 0, 0, 0.05, 0.01, 0, 0, 0.07]))
        linkCollisionShapeIndices.extend(
            [joint_c, joint_c, right_shoulders_c, joint_c, joint_c, left_shoulders_c, neck_c, joint_c, joint_c,
             head_c])
        linkVisualShapeIndices.extend(
            [joint_v, joint_v, right_shoulders_v, joint_v, joint_v, left_shoulders_v, neck_v, joint_v, joint_v,
             head_v])
        linkPositions.extend(
            [shoulders_p, shoulders_p, joint_p, shoulders_p, shoulders_p, joint_p, neck_p, head_p, joint_p,
             joint_p])
        linkOrientations.extend([joint_o] * 10)
        linkInertialFramePositions.extend([[0, 0, 0]] * 10)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 10)
        linkParentIndices.extend([0, 1, 2, 0, 4, 5, 0, 7, 8, 9])  # joint的连接顺序
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 10)
        linkJointAxis.extend(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0],
             [0, 0, 1]])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(-10), np.deg2rad(-10), np.deg2rad(-35), np.deg2rad(-10), np.deg2rad(-30), np.deg2rad(-35),
             np.deg2rad(-10), np.deg2rad(-50), np.deg2rad(-34), np.deg2rad(-70)]) * limit_scale)
        linkUpperLimits.extend(np.array(
            [np.deg2rad(10), np.deg2rad(30), np.deg2rad(35), np.deg2rad(10), np.deg2rad(10), np.deg2rad(35),
             np.deg2rad(20), np.deg2rad(50), np.deg2rad(34), np.deg2rad(70)]) * limit_scale)

        # NOTE: Right arm
        linkMasses.extend(m * np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))

        linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        linkPositions.extend([right_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o] * 7)
        linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        linkParentIndices.extend([3, 11, 12, 13, 14, 15, 16])
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(5), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-81),
             np.deg2rad(-27)]) * limit_scale)
        linkUpperLimits.extend(np.array(
            [np.deg2rad(198), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90),
             np.deg2rad(47)]) * limit_scale)

        # NOTE: Left arm
        linkMasses.extend(m * np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        linkPositions.extend([left_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o] * 7)
        linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        linkParentIndices.extend([6, 18, 19, 20, 21, 22, 23])
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(-198), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90),
             np.deg2rad(-81), np.deg2rad(-47)]) * limit_scale)
        linkUpperLimits.extend(np.array(
            [np.deg2rad(-5), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90),
             np.deg2rad(27)]) * limit_scale)

        # NOTE: Waist and hips
        linkMasses.extend(m * np.array([0, 0, 0.13, 0.14]))
        linkCollisionShapeIndices.extend([waist_c, joint_c, joint_c, hips_c])
        linkVisualShapeIndices.extend([waist_v, joint_v, joint_v, hips_v])
        linkPositions.extend([waist_p, hips_p, joint_p, joint_p])
        linkOrientations.extend([joint_o] * 4)
        linkInertialFramePositions.extend([[0, 0, 0]] * 4)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 4)
        linkParentIndices.extend([0, 25, 26, 27])
        linkJointTypes.extend([p.JOINT_FIXED] + [p.JOINT_REVOLUTE] * 3)
        linkJointAxis.extend([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array([0, np.deg2rad(-75), np.deg2rad(-30), np.deg2rad(-30)]))
        linkUpperLimits.extend(np.array([0, np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)]))

        # NOTE: Right leg
        linkMasses.extend(m * np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        linkPositions.extend([right_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        linkOrientations.extend([joint_o] * 7)
        linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        linkParentIndices.extend([28, 29, 30, 31, 32, 33, 34])
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        linkJointAxis.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(-127), np.deg2rad(-40), np.deg2rad(-45), 0, np.deg2rad(-35), np.deg2rad(-23),
             np.deg2rad(-43)]))
        linkUpperLimits.extend(np.array(
            [np.deg2rad(30), np.deg2rad(45), np.deg2rad(40), np.deg2rad(130), np.deg2rad(38), np.deg2rad(24),
             np.deg2rad(35)]))

        # NOTE: Left leg
        linkMasses.extend(m * np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        linkPositions.extend([left_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        linkOrientations.extend([joint_o] * 7)
        linkInertialFramePositions.extend([[0, 0, 0]] * 7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]] * 7)
        linkParentIndices.extend([28, 36, 37, 38, 39, 40, 41])
        linkJointTypes.extend([p.JOINT_REVOLUTE] * 7)
        linkJointAxis.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array(
            [np.deg2rad(-127), np.deg2rad(-45), np.deg2rad(-40), 0, np.deg2rad(-35), np.deg2rad(-24),
             np.deg2rad(-35)]))
        linkUpperLimits.extend(np.array(
            [np.deg2rad(30), np.deg2rad(40), np.deg2rad(45), np.deg2rad(130), np.deg2rad(38), np.deg2rad(23),
             np.deg2rad(43)]))

        human = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=chest_c,
                                  baseVisualShapeIndex=chest_v, basePosition=chest_p, baseOrientation=[0, 0, 0, 1],
                                  linkMasses=linkMasses, linkCollisionShapeIndices=linkCollisionShapeIndices,
                                  linkVisualShapeIndices=linkVisualShapeIndices, linkPositions=linkPositions,
                                  linkOrientations=linkOrientations,
                                  linkInertialFramePositions=linkInertialFramePositions,
                                  linkInertialFrameOrientations=linkInertialFrameOrientations,
                                  linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes,
                                  linkJointAxis=linkJointAxis, linkLowerLimits=linkLowerLimits,
                                  linkUpperLimits=linkUpperLimits, useMaximalCoordinates=False,
                                  flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        num_joints = p.getNumJoints(human, physicsClientId=self.id)  # 移除碰撞
        for i in range(-1, num_joints):
            for j in range(-1, num_joints):
                p.setCollisionFilterPair(human, human, i, j, 0, physicsClientId=self.id)
        for i in range(3, 10):  # Right arm
            for j in [-1] + list(range(10, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(13, 20):  # Left arm
            for j in list(range(-1, 10)) + list(range(20, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(28, 35):  # Right leg
            for j in list(range(-1, 24)) + list(range(35, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(35, num_joints):  # Left leg
            for j in list(range(-1, 24)) + list(range(28, 35)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)

        # Enforce joint limits
        human_joint_positions = get_pos(human,
                                        list(range(p.getNumJoints(human, physicsClientId=self.id))),self.id)  # 可以在外使用force所以可删除
        for j in range(p.getNumJoints(human, physicsClientId=self.id)):  # 增加state限制
            joint_info = p.getJointInfo(human, j, physicsClientId=self.id)
            joint_pos = human_joint_positions[j]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if joint_pos < lower_limit:
                p.resetJointState(human, jointIndex=j, targetValue=lower_limit, targetVelocity=0,
                                  physicsClientId=self.id)
            elif joint_pos > upper_limit:
                p.resetJointState(human, jointIndex=j, targetValue=upper_limit, targetVelocity=0,
                                  physicsClientId=self.id)
        for i in [-1,2,5,12,15,24]:
            p.changeVisualShape(human, i, rgbaColor=[0.85, 0.3, 0.1, 1], physicsClientId=self.id)
        for i in [27,30,37]:
            p.changeVisualShape(human, i, rgbaColor=[0.2, 0.3, 0.8, 1], physicsClientId=self.id)
        return human

    def show(self):
        pass

    def createwheelchair(self):
        wheelchair = p.loadURDF(os.path.join(self.directory, 'wheelchair', 'wheelchair.urdf'),
                                physicsClientId=self.id)  # 改为BLACK？
        # Initialize chair position
        p.resetBasePositionAndOrientation(wheelchair, [0, 0, 0.06],
                                          p.getQuaternionFromEuler([np.pi / 2.0, 0, np.pi], physicsClientId=self.id),
                                          physicsClientId=self.id)
        p.changeVisualShape(wheelchair , -1, rgbaColor=[0.6, 0.6, 0.6, 1.0], physicsClientId=self.id)  # 头
        return wheelchair

    def createplane(self):
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'),
                   physicsClientId=self.id)

    def createrobot(self):
        robot = p.loadURDF(os.path.join(self.directory, 'PR2', 'pr2_no_torso_lift_tall.urdf'),
                           useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_INERTIA_FROM_FILE,
                           physicsClientId=self.id)
        # Initialize and position PR2
        p.resetBasePositionAndOrientation(robot, [-2, -2, 0], [0, 0, 0, 1], physicsClientId=self.id)
        for i in [19, 42, 64, 43, 46, 49, 58, 60, 65, 68, 71, 80, 82,45, 51, 67, 73,40]:
            p.changeVisualShape(robot, i, rgbaColor=[0.4, 0.4, 0.4, 1], physicsClientId=self.id)
        p.changeVisualShape(robot, 20, rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=self.id)#头

        return robot

    def create_body(self, shape=p.GEOM_CAPSULE,  radius=0,
                    length=0, position_offset=[0, 0, 0], orientation=[0, 0, 0, 1],specular_color=[0.1, 0.1, 0.1], rgba_Color=[0.8, 0.6, 0.4, 1],):  # 给身体不同部分创建不同颜色
        visual_shape = p.createVisualShape(shape, radius=radius, length=length, rgbaColor=rgba_Color,
                                           specularColor=specular_color, visualFramePosition=position_offset,
                                           visualFrameOrientation=orientation, physicsClientId=self.id)  # 创建反射
        collision_shape = p.createCollisionShape(shape, radius=radius, height=length,
                                                 collisionFramePosition=position_offset,
                                                 collisionFrameOrientation=orientation, physicsClientId=self.id)
        # return uniqel id
        return collision_shape, visual_shape



    def create_table(self):
        table = p.loadURDF(os.path.join(self.directory, 'table', 'table_tall.urdf'),
                           basePosition=[0.35, -0.9, 0],
                           baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id),
                           physicsClientId=self.id)
        return table

    def create_bowl(self):
        bowl_scale = 0.75
        visual_filename = os.path.join(self.directory, 'dinnerware',
                                       'bowl_reduced_compressed.obj')  # 重排列文件设置使导入更容易
        collision_filename = os.path.join(self.directory, 'dinnerware', 'bowl_vhacd.obj')
        bowl_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename,
                                          meshScale=[bowl_scale] * 3, physicsClientId=self.id)
        bowl_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename,
                                                meshScale=[bowl_scale] * 3, physicsClientId=self.id)
        bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array(
            [self.np_random.uniform(-0.05, 0.05), self.np_random.uniform(-0.05, 0.05), 0])
        bowl = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=bowl_collision,
                                 baseVisualShapeIndex=bowl_visual, basePosition=bowl_pos,
                                 baseOrientation=p.getQuaternionFromEuler([np.pi / 2.0, 0, 0],
                                                                          physicsClientId=self.id),
                                 baseInertialFramePosition=[0, 0.04 * bowl_scale, 0],
                                 useMaximalCoordinates=False, physicsClientId=self.id)

        return bowl

    def create_target(self, human):
        # Set target on mouth
        mouth_pos = [0, -0.11, 0.03]
        head_pos, head_orient = p.getLinkState(human, 23, computeForwardKinematics=True, physicsClientId=self.id)[
                                :2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, mouth_pos, [0, 0, 0, 1],
                                                         physicsClientId=self.id)  # 将mouth位置转换到相对头的位置
        target_pos = np.array(target_pos)
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1],
                                            physicsClientId=self.id)
        # target
        target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                   baseVisualShapeIndex=sphere_visual, basePosition=target_pos,
                                   useMaximalCoordinates=False, physicsClientId=self.id)
        p.changeVisualShape(target, -1, rgbaColor=[0.9, 0.3, 0.3, 1], physicsClientId=self.id)
        return target, target_pos

    def create_food(self, spoon):
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(spoon, physicsClientId=self.id)
        spoon_pos = np.array(spoon_pos)
        food_radius = 0.005
        food_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=food_radius, physicsClientId=self.id)
        food_mass = 0.001
        food_count = 2 * 2 * 2
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i * 2 * food_radius - 0.005, j * 2 * food_radius,
                                                     k * 2 * food_radius + 0.02]) + spoon_pos)  # 初始化到不同的位置
        last_food_id = p.createMultiBody(baseMass=food_mass, baseCollisionShapeIndex=food_collision,
                                         baseVisualShapeIndex=-1, basePosition=[0, 0, 0], useMaximalCoordinates=False,
                                         batchPositions=batch_positions, physicsClientId=self.id)
        foods = list(range(last_food_id - food_count + 1, last_food_id + 1))
        return foods

    def create_spoon(self, robot, mesh_scale=[1] * 3, pos_offset=[0] * 3, orient_offset=[0, 0, 0, 1]):
        gripper_pos, gripper_orient = p.getLinkState(robot, 54, computeForwardKinematics=True, physicsClientId=self.id)[
                                      :2]
        transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient,
                                                               positionB=pos_offset, orientationB=orient_offset,
                                                               physicsClientId=self.id)
        visual_filename = os.path.join(self.directory, 'dinnerware', 'spoon_reduced_compressed.obj')
        collision_filename = os.path.join(self.directory, 'dinnerware', 'spoon_vhacd.obj')
        spoon_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale,
                                           rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
        spoon_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename,
                                                 meshScale=mesh_scale, physicsClientId=self.id)
        spoon = p.createMultiBody(baseMass=0.01, baseCollisionShapeIndex=spoon_collision,
                                  baseVisualShapeIndex=spoon_visual, basePosition=transform_pos,
                                  baseOrientation=transform_orient, useMaximalCoordinates=False,
                                  physicsClientId=self.id)
        for j in range(49, 64):
            for tj in list(range(p.getNumJoints(spoon, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(robot, spoon, j, tj, False, physicsClientId=self.id)
        constraint = p.createConstraint(robot, 54, spoon, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=pos_offset,
                                        childFramePosition=[0, 0, 0], parentFrameOrientation=orient_offset,
                                        physicsClientId=self.id)
        p.changeConstraint(constraint, maxForce=500, physicsClientId=self.id)  # 连接物体

        return spoon
