from common.path_config import PROJECT_ROOT

import numpy as np
from common.utils import FSMCommand


class StateAndCmd:
    def __init__(self, num_joints):
        # robot state
        self.num_joints = num_joints
        self.q = np.zeros(num_joints, dtype=np.float32)
        self.dq = np.zeros(num_joints, dtype=np.float32)
        self.ddq = np.zeros(num_joints, dtype=np.float32)
        self.tau_est = np.zeros(num_joints, dtype=np.float32)
        self.gravity_ori = np.array([0., 0., 1.])
        self.ang_vel = np.zeros(3)
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.base_quat = np.zeros(4, dtype=np.float32)
        self.base_pos = np.zeros(3, dtype=np.float32)
        self.obj_pos = np.zeros(3, dtype=np.float32)
        self.obj_quat = np.zeros(4, dtype=np.float32)
        self.ball_pos = np.zeros(3, dtype=np.float32)
        self.ball_quat = np.zeros(4, dtype=np.float32)
        self.push_box_pos = np.zeros(3, dtype=np.float32)
        self.push_box_quat = np.zeros(4, dtype=np.float32)
        self.carry_box_pos = np.zeros(3, dtype=np.float32)
        self.carry_box_quat = np.zeros(4, dtype=np.float32)
        self.stack_box_pos = np.zeros((3, 3), dtype=np.float32)
        self.stack_box_quat = np.zeros((3, 4), dtype=np.float32)
        self.skill_cmd = FSMCommand.INVALID

class PolicyOutput:
    def __init__(self, num_joints):
        # actions
        self.actions = np.zeros(num_joints, dtype=np.float32)
        self.kps = np.zeros(num_joints, dtype=np.float32)
        self.kds = np.zeros(num_joints, dtype=np.float32)
        self.target = np.zeros(3+4+num_joints+3+4, dtype=np.float32)
        self.wrist_goal = np.zeros(7+7, dtype=np.float32)
        self.contact = np.zeros(4, dtype=np.float32) # contact flags (left_foot, right_foot, left_hand, right_hand)
        # reference visualization (pos + wxyz quat)
        self.torso_goal = np.zeros(7, dtype=np.float32)
        self.l_ankle_goal = np.zeros(7, dtype=np.float32)
        self.r_ankle_goal = np.zeros(7, dtype=np.float32)
