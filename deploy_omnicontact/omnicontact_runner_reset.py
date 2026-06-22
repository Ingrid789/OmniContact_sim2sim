import mujoco
import numpy as np

from common.utils import get_gravity_orientation, yaw_to_quat
from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ
from omnicontact_runner_utils import sample_xy_around


class OmniContactResetMixin:
    _LOCO_CMD_RANGE = np.array(
        [
            [-0.4, 0.7],
            [-0.4, 0.4],
            [-1.57, 1.57],
        ],
        dtype=np.float32,
    )
    _LOCO_MAX_LIN_SPEED = 0.45
    _LOCO_MAX_YAW_RATE = 0.8
    _LOCO_POS_TOLERANCE = 0.08
    _LOCO_YAW_TOLERANCE = 0.08

    def _normalize_vel_cmd(self, target_cmd: np.ndarray) -> np.ndarray:
        cmd = np.asarray(target_cmd, dtype=np.float32).reshape(3)
        ranges = self._LOCO_CMD_RANGE
        normalized = 2.0 * (cmd - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _update_vel_cmd(self):
        target_cmd = np.zeros(3, dtype=np.float32)
        if getattr(self.policy, "task", "") == "loco":
            goal = self._goal_pos()
            delta_world = goal[:2] - self.state_cmd.base_pos[:2]
            dist = float(np.linalg.norm(delta_world))
            if dist > self._LOCO_POS_TOLERANCE:
                base_quat = np.asarray(self.state_cmd.base_quat, dtype=np.float32).reshape(4)
                yaw = float(np.arctan2(
                    2.0 * (base_quat[0] * base_quat[3] + base_quat[1] * base_quat[2]),
                    1.0 - 2.0 * (base_quat[2] * base_quat[2] + base_quat[3] * base_quat[3]),
                ))
                c = float(np.cos(yaw))
                s = float(np.sin(yaw))
                delta_body = np.array(
                    [
                        c * delta_world[0] + s * delta_world[1],
                        -s * delta_world[0] + c * delta_world[1],
                    ],
                    dtype=np.float32,
                )
                direction_body = delta_body / max(float(np.linalg.norm(delta_body)), 1e-6)
                speed = min(self._LOCO_MAX_LIN_SPEED, dist)
                target_cmd[:2] = direction_body * speed

                target_yaw = float(np.arctan2(delta_world[1], delta_world[0]))
                yaw_error = float(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw)))
                if abs(yaw_error) > self._LOCO_YAW_TOLERANCE:
                    target_cmd[2] = float(np.clip(yaw_error, -self._LOCO_MAX_YAW_RATE, self._LOCO_MAX_YAW_RATE))
        self.state_cmd.vel_cmd = self._normalize_vel_cmd(target_cmd)

    def _sync_state_cmd_from_mj(self):
        self.state_cmd.q = self.d.qpos[7 : 7 + self.num_joints].copy()
        self.state_cmd.dq = self.d.qvel[6 : 6 + self.num_joints].copy()
        self.state_cmd.base_pos = self.d.qpos[:3].copy()
        self.state_cmd.base_quat = self.d.qpos[3:7].copy()
        self.state_cmd.lin_vel = self.d.qvel[:3].copy()
        self.state_cmd.ang_vel = self.d.qvel[3:6].copy()
        self.state_cmd.gravity_ori = get_gravity_orientation(self.state_cmd.base_quat).astype(np.float32)

        if self.push_box_body_id >= 0:
            self.state_cmd.push_box_pos = self.d.xpos[self.push_box_body_id].copy()
            self.state_cmd.push_box_quat = self.d.xquat[self.push_box_body_id].copy()
        if self.ball_body_id >= 0:
            self.state_cmd.ball_pos = self.d.xpos[self.ball_body_id].copy()
            self.state_cmd.ball_quat = self.d.xquat[self.ball_body_id].copy()
        if self.carry_box_body_id >= 0:
            self.state_cmd.carry_box_pos = self.d.xpos[self.carry_box_body_id].copy()
            self.state_cmd.carry_box_quat = self.d.xquat[self.carry_box_body_id].copy()
        for i, body_id in enumerate(self.stack_box_body_ids):
            if int(body_id) >= 0:
                self.state_cmd.stack_box_pos[i] = self.d.xpos[int(body_id)].copy()
                self.state_cmd.stack_box_quat[i] = self.d.xquat[int(body_id)].copy()

        active_body_id = self._get_active_object_body_id()
        if active_body_id >= 0:
            self.state_cmd.obj_pos = self.d.xpos[active_body_id].copy()
            self.state_cmd.obj_quat = self.d.xquat[active_body_id].copy()
        self._update_vel_cmd()

    def _reset_env_cfgen(self):
        if self.is_carryheart:
            self._reset_carryheart_env()
            return

        init_pos = self._init_pos()
        if self.policy.task == "loco":
            self.d.qpos[:3] = init_pos
            self.d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.d.qpos[7 : 7 + self.num_joints] = DEFAULT_JOINT_POS_MJ[: self.num_joints]
            self.d.qvel[: 6 + self.num_joints] = 0.0
            if self.plane1_mocap_id >= 0:
                self.d.mocap_pos[self.plane1_mocap_id] = init_pos
            if self.plane2_mocap_id >= 0:
                self.d.mocap_pos[self.plane2_mocap_id] = self._goal_pos()
            mujoco.mj_forward(self.m, self.d)
            return

        object_qpos_adr = self._get_active_object_qpos_adr()
        object_qvel_adr = self._get_active_object_qvel_adr()
        if object_qpos_adr >= 0:
            self.d.qpos[object_qpos_adr : object_qpos_adr + 3] = init_pos
            self.d.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if object_qvel_adr >= 0:
            self.d.qvel[object_qvel_adr : object_qvel_adr + 6] = 0.0
        if self.policy.task == "push-carry" and self.carry_box_qpos_adr >= 0:
            carry_half_z = float(getattr(self.policy, "carry_box_dims", np.array([0.15, 0.15, 0.15], dtype=np.float32))[2])
            override = getattr(self.policy, "carry_box_init_pos_override", None)
            if override is None:
                carry_init_pos = np.array(
                    [
                        float(np.random.uniform(-5.0, 5.0)),
                        float(np.random.uniform(-5.0, 5.0)),
                        carry_half_z,
                    ],
                    dtype=np.float32,
                )
            else:
                carry_init_pos = np.asarray(override, dtype=np.float32).reshape(3).copy()
                carry_init_pos[2] = carry_half_z
            self.d.qpos[self.carry_box_qpos_adr : self.carry_box_qpos_adr + 3] = carry_init_pos
            self.d.qpos[self.carry_box_qpos_adr + 3 : self.carry_box_qpos_adr + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            if self.carry_box_qvel_adr >= 0:
                self.d.qvel[self.carry_box_qvel_adr : self.carry_box_qvel_adr + 6] = 0.0
        if self.policy.task == "carry-push" and self.push_box_joint_id >= 0:
            push_half_z = float(getattr(self.policy, "push_box_dims", np.array([0.23, 0.25, 0.26], dtype=np.float32))[2])
            override = getattr(self.policy, "carry_box_init_pos_override", None)
            if override is None:
                push_init_pos = np.array(
                    [
                        float(np.random.uniform(-5.0, 5.0)),
                        float(np.random.uniform(-5.0, 5.0)),
                        push_half_z,
                    ],
                    dtype=np.float32,
                )
            else:
                push_init_pos = np.asarray(override, dtype=np.float32).reshape(3).copy()
                push_init_pos[2] = push_half_z
            push_box_qpos_adr = int(self.m.jnt_qposadr[self.push_box_joint_id])
            push_box_qvel_adr = int(self.m.jnt_dofadr[self.push_box_joint_id])
            self.d.qpos[push_box_qpos_adr : push_box_qpos_adr + 3] = push_init_pos
            self.d.qpos[push_box_qpos_adr + 3 : push_box_qpos_adr + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            self.d.qvel[push_box_qvel_adr : push_box_qvel_adr + 6] = 0.0
        if self.policy.task == "push-relocate" and self.ball_qpos_adr >= 0:
            ball_half_z = float(getattr(self.policy, "ball_dims", np.array([0.10, 0.10, 0.10], dtype=np.float32))[2])
            override = getattr(self.policy, "ball_init_pos_override", None)
            if override is None:
                ball_init_pos = np.array(
                    [
                        float(np.random.uniform(-5.0, 5.0)),
                        float(np.random.uniform(-5.0, 5.0)),
                        ball_half_z,
                    ],
                    dtype=np.float32,
                )
            else:
                ball_init_pos = np.asarray(override, dtype=np.float32).reshape(3).copy()
                ball_init_pos[2] = ball_half_z
            self.d.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3] = ball_init_pos
            self.d.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            if self.ball_qvel_adr >= 0:
                self.d.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = 0.0
        if self.policy.task in {"stackbox", "carry-carry", "carry-carry-carry"}:
            stack_dims = np.asarray(
                getattr(
                    self.policy,
                    "stack_box_dims",
                    np.array([[0.20, 0.20, 0.15], [0.15, 0.15, 0.15], [0.10, 0.10, 0.10]], dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(3, 3)
            stack_count = int(getattr(self.policy, "stackbox_stage_count", 3))
            for i, qpos_adr in enumerate(self.stack_box_qpos_adrs[:stack_count]):
                if int(qpos_adr) < 0:
                    continue
                init_xy = sample_xy_around(
                    np.array([1.0, 0.0], dtype=np.float32),
                    1.0,
                    5.0,
                    y_positive=True,
                    angle_range_deg=120.0,
                )
                init_pos = np.array([init_xy[0], init_xy[1], stack_dims[i, 2]], dtype=np.float32)
                self.d.qpos[int(qpos_adr) : int(qpos_adr) + 3] = init_pos
                self.d.qpos[int(qpos_adr) + 3 : int(qpos_adr) + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                qvel_adr = int(self.stack_box_qvel_adrs[i])
                if qvel_adr >= 0:
                    self.d.qvel[qvel_adr : qvel_adr + 6] = 0.0

        if self.plane1_mocap_id >= 0:
            self.d.mocap_pos[self.plane1_mocap_id] = init_pos - np.array([0.0, 0.0, self._table_height_offset() + 0.01], dtype=np.float32)
        if self.plane2_mocap_id >= 0:
            if self.policy.task in {"stackbox", "carry-carry", "carry-carry-carry"}:
                self.d.mocap_pos[self.plane2_mocap_id] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif self.policy.task == "kickball":
                goal_pos = self._goal_pos()
                self.d.mocap_pos[self.plane2_mocap_id] = np.array(
                    [goal_pos[0], goal_pos[1], 0.0],
                    dtype=np.float32,
                )
                goal_dir = (goal_pos[:2] - init_pos[:2]).astype(np.float32)
                goal_dir_norm = float(np.linalg.norm(goal_dir))
                if goal_dir_norm > 1e-6:
                    goal_yaw = float(np.arctan2(float(goal_dir[1]), float(goal_dir[0])))
                    self.d.mocap_quat[self.plane2_mocap_id] = yaw_to_quat(goal_yaw).astype(np.float32)
            elif self.policy.task == "push-relocate":
                goal_pos = self._goal_pos()
                self.d.mocap_pos[self.plane2_mocap_id] = np.array(
                    [goal_pos[0], goal_pos[1], -0.01],
                    dtype=np.float32,
                )
            else:
                goal_pos = self._plane2_goal_pos()
                self.d.mocap_pos[self.plane2_mocap_id] = goal_pos - np.array([0.0, 0.0, self._table_height_offset() + 0.01], dtype=np.float32
                )

        mujoco.mj_step(self.m, self.d)

    def _reset_env_tracking_npz(self):
        if not hasattr(self.policy, "ref_object_pos"):
            return

        if self.policy.reference_source == "NPZmotion" and hasattr(self.policy, "ref_base_pos"):
            self.d.qpos[:3] = self.policy.ref_base_pos[0]
            self.d.qpos[3:7] = self.policy.ref_base_quat[0]
            self.d.qpos[7 : 7 + self.num_joints] = self.policy.ref_joint_pos[0, self.policy.lab2mj]

        object_qpos_adr = self._get_active_object_qpos_adr()
        object_qvel_adr = self._get_active_object_qvel_adr()
        if object_qpos_adr >= 0:
            self.d.qpos[object_qpos_adr : object_qpos_adr + 3] = self.policy.ref_object_pos[0]
            self.d.qpos[object_qpos_adr + 3 : object_qpos_adr + 7] = self.policy.ref_object_quat[0]
        if hasattr(self.policy, "ref_object_lin_vel") and hasattr(self.policy, "ref_object_ang_vel"):
            if object_qvel_adr >= 0:
                self.d.qvel[object_qvel_adr : object_qvel_adr + 3] = self.policy.ref_object_lin_vel[0]
                self.d.qvel[object_qvel_adr + 3 : object_qvel_adr + 6] = self.policy.ref_object_ang_vel[0]
        elif object_qvel_adr >= 0:
            self.d.qvel[object_qvel_adr : object_qvel_adr + 6] = 0.0

        if self.plane1_mocap_id >= 0 and hasattr(self.policy, "ref_table_1_pos"):
            self.d.mocap_pos[self.plane1_mocap_id] = self.policy.ref_table_1_pos[0]
        if self.plane2_mocap_id >= 0 and hasattr(self.policy, "ref_table_2_pos"):
            self.d.mocap_pos[self.plane2_mocap_id] = self.policy.ref_table_2_pos[0]
        mujoco.mj_forward(self.m, self.d)

    def _prepare_episode(self):
        self._reset_episode_metrics()
        self.replan.reset_episode()
        if self.policy.reference_source == "CFgen" and self.args.reset_env:
            self._reset_env_cfgen()

        self._sync_state_cmd_from_mj()
        self.policy.enter()

        if self.policy.reference_source == "NPZmotion" and self.args.reset_env:
            self._reset_env_tracking_npz()

        self._sync_state_cmd_from_mj()
