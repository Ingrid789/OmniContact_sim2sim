import mujoco
import numpy as np


class OmniContactVisualizationMixin:
    def _name2id(self, obj_type: int, name: str) -> int:
        return int(mujoco.mj_name2id(self.m, obj_type, name))

    def _safe_body_mocap_id(self, body_name: str) -> int:
        body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return -1
        return int(self.m.body_mocapid[body_id])

    def _first_existing_name(self, obj_type: int, candidates: tuple[str, ...]) -> tuple[int, str | None]:
        for name in candidates:
            obj_id = self._name2id(obj_type, name)
            if obj_id >= 0:
                return obj_id, name
        return -1, None

    def _cache_visual_ids(self):
        self.object_body_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_BODY,
            ("box", "ball"),
        )
        self.object_geom_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_GEOM,
            ("box_geom", "ball_geom"),
        )
        self.object_joint_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            ("box", "ball"),
        )
        self.push_box_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "push_box")
        self.push_box_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "push_box_geom_top")
        self.push_box_joint_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            ("push_box", "box"),
        )
        self.ball_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.ball_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "ball")
        self.ball_qpos_adr = int(self.m.jnt_qposadr[self.ball_joint_id]) if self.ball_joint_id >= 0 else -1
        self.ball_qvel_adr = int(self.m.jnt_dofadr[self.ball_joint_id]) if self.ball_joint_id >= 0 else -1
        self.carry_box_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "carry_box")
        self.carry_box_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "carry_box_geom")
        self.carry_box_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "carry_box_joint")
        self.carry_box_qpos_adr = int(self.m.jnt_qposadr[self.carry_box_joint_id]) if self.carry_box_joint_id >= 0 else -1
        self.carry_box_qvel_adr = int(self.m.jnt_dofadr[self.carry_box_joint_id]) if self.carry_box_joint_id >= 0 else -1
        self.stack_box_names = ("stack_box_large", "stack_box_mid", "stack_box_small")
        self.stack_box_joint_names = ("stack_box_large_joint", "stack_box_mid_joint", "stack_box_small_joint")
        self.stack_box_geom_names = ("stack_box_large_geom", "stack_box_mid_geom", "stack_box_small_geom")
        self.stack_box_ghost_joint_names = (
            "ghost_stack_box_large_joint",
            "ghost_stack_box_mid_joint",
            "ghost_stack_box_small_joint",
        )
        self.stack_box_body_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_BODY, name) for name in self.stack_box_names],
            dtype=np.int32,
        )
        self.stack_box_geom_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.stack_box_geom_names],
            dtype=np.int32,
        )
        self.stack_box_joint_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.stack_box_joint_names],
            dtype=np.int32,
        )
        self.stack_box_qpos_adrs = np.array(
            [int(self.m.jnt_qposadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.stack_box_joint_ids],
            dtype=np.int32,
        )
        self.stack_box_qvel_adrs = np.array(
            [int(self.m.jnt_dofadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.stack_box_joint_ids],
            dtype=np.int32,
        )
        self.stack_box_ghost_joint_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.stack_box_ghost_joint_names],
            dtype=np.int32,
        )
        self.stack_box_ghost_qpos_adrs = np.array(
            [int(self.m.jnt_qposadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.stack_box_ghost_joint_ids],
            dtype=np.int32,
        )
        self.stack_box_ghost_qvel_adrs = np.array(
            [int(self.m.jnt_dofadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.stack_box_ghost_joint_ids],
            dtype=np.int32,
        )
        self.carryheart_box_names = tuple(["box"] + [f"box_{i}" for i in range(2, 11)])
        self.carryheart_box_joint_names = tuple(["box"] + [f"box_{i}_joint" for i in range(2, 11)])
        self.carryheart_box_geom_names = tuple(["box_geom"] + [f"box_{i}_geom" for i in range(2, 11)])
        self.carryheart_box_body_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_BODY, name) for name in self.carryheart_box_names],
            dtype=np.int32,
        )
        self.carryheart_box_geom_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.carryheart_box_geom_names],
            dtype=np.int32,
        )
        self.carryheart_box_joint_ids = np.array(
            [self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.carryheart_box_joint_names],
            dtype=np.int32,
        )
        self.carryheart_box_qpos_adrs = np.array(
            [int(self.m.jnt_qposadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.carryheart_box_joint_ids],
            dtype=np.int32,
        )
        self.carryheart_box_qvel_adrs = np.array(
            [int(self.m.jnt_dofadr[joint_id]) if joint_id >= 0 else -1 for joint_id in self.carryheart_box_joint_ids],
            dtype=np.int32,
        )
        self.heart_segment_mocap_ids = [
            self._safe_body_mocap_id(f"heart_seg_{i}_holder") for i in range(1, len(self._HEART_SEGMENT_GOAL_PAIRS) + 1)
        ]
        self.heart_segment_geom_ids = [
            self._name2id(mujoco.mjtObj.mjOBJ_GEOM, f"heart_seg_{i}") for i in range(1, len(self._HEART_SEGMENT_GOAL_PAIRS) + 1)
        ]
        self.ghost_object_joint_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            ("ghost_push_box_joint", "ghost_box_joint", "ghost_ball_joint"),
        )
        self.ghost_object_qpos_adr = int(self.m.jnt_qposadr[self.ghost_object_joint_id]) if self.ghost_object_joint_id >= 0 else -1
        self.ghost_object_qvel_adr = int(self.m.jnt_dofadr[self.ghost_object_joint_id]) if self.ghost_object_joint_id >= 0 else -1
        self.ghost_carry_box_joint_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            ("ghost_carry_box_joint",),
        )
        self.ghost_carry_box_qpos_adr = int(self.m.jnt_qposadr[self.ghost_carry_box_joint_id]) if self.ghost_carry_box_joint_id >= 0 else -1
        self.ghost_carry_box_qvel_adr = int(self.m.jnt_dofadr[self.ghost_carry_box_joint_id]) if self.ghost_carry_box_joint_id >= 0 else -1
        self.ghost_ball_joint_id, _ = self._first_existing_name(
            mujoco.mjtObj.mjOBJ_JOINT,
            ("ghost_ball_joint",),
        )
        self.ghost_ball_qpos_adr = int(self.m.jnt_qposadr[self.ghost_ball_joint_id]) if self.ghost_ball_joint_id >= 0 else -1
        self.ghost_ball_qvel_adr = int(self.m.jnt_dofadr[self.ghost_ball_joint_id]) if self.ghost_ball_joint_id >= 0 else -1
        self.ghost_robot_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "ghost_floating_base_joint")
        self.ghost_robot_qpos_adr = int(self.m.jnt_qposadr[self.ghost_robot_joint_id]) if self.ghost_robot_joint_id >= 0 else -1
        self.ghost_robot_qvel_adr = int(self.m.jnt_dofadr[self.ghost_robot_joint_id]) if self.ghost_robot_joint_id >= 0 else -1
        self.ghost_robot_joint_qpos_adrs = np.array(
            [
                int(self.m.jnt_qposadr[joint_id]) if joint_id >= 0 else -1
                for joint_id in (
                    self._name2id(mujoco.mjtObj.mjOBJ_JOINT, f"ghost_{name}")
                    for name in self.policy.kinematics.joint_names
                )
            ],
            dtype=np.int32,
        )

        self.l_wrist_mocap_id = self._safe_body_mocap_id("ref_l_wrist_frame")
        self.r_wrist_mocap_id = self._safe_body_mocap_id("ref_r_wrist_frame")
        self.torso_mocap_id = self._safe_body_mocap_id("ref_torso_frame")
        self.l_ankle_mocap_id = self._safe_body_mocap_id("ref_l_ankle_frame")
        self.r_ankle_mocap_id = self._safe_body_mocap_id("ref_r_ankle_frame")
        self.plane1_mocap_id = self._safe_body_mocap_id("plane_1_holder")
        self.plane2_mocap_id = self._safe_body_mocap_id("plane_2_holder")

        self.ref_l_hand_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ref_l_rubber_hand")
        self.ref_r_hand_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ref_r_rubber_hand")
        self.ref_l_ankle_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ref_l_ankle_mesh")
        self.ref_r_ankle_geom_id = self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "ref_r_ankle_mesh")

        self.torso_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.left_palm_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_palm_link")
        self.right_palm_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_palm_link")
        self.left_wrist_yaw_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.right_wrist_yaw_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.left_hand_collision_geoms = self._body_geom_ids(self.left_wrist_yaw_body_id)
        self.right_hand_collision_geoms = self._body_geom_ids(self.right_wrist_yaw_body_id)
        self.left_ankle_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link")
        self.right_ankle_body_id = self._name2id(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_pitch_link")

    def _body_geom_ids(self, body_id: int) -> list[int]:
        if body_id < 0:
            return []
        geom_num = int(self.m.body_geomnum[body_id])
        geom_adr = int(self.m.body_geomadr[body_id])
        return [geom_adr + i for i in range(geom_num)]

    def _set_mocap_pose(self, mocap_id: int, pose7: np.ndarray):
        if mocap_id < 0:
            return
        self.d.mocap_pos[mocap_id] = pose7[:3]
        self.d.mocap_quat[mocap_id] = pose7[3:7]

    def _set_freejoint_pose(self, qpos_adr: int, qvel_adr: int, pose7: np.ndarray):
        if qpos_adr < 0:
            return
        pose7 = np.asarray(pose7, dtype=np.float32).reshape(7)
        self.d.qpos[qpos_adr : qpos_adr + 7] = pose7
        if qvel_adr >= 0:
            self.d.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _update_ghost_robot_visualization(self):
        if self.ghost_robot_qpos_adr < 0:
            return

        dof_pos = getattr(self.policy, "dof_pos", None)
        if dof_pos is not None:
            dof_pos = np.asarray(dof_pos, dtype=np.float32)
        else:
            ref_joint_pos = getattr(self.policy, "ref_joint_pos", None)
            if ref_joint_pos is None:
                return

            dof_pos = np.asarray(ref_joint_pos, dtype=np.float32)
            if dof_pos.ndim != 2:
                return

            # tracking_npz joint_pos is stored in Lab order; ghost XML joints are
            # addressed by MuJoCo joint names, same as the main robot qpos order.
            lab2mj = getattr(self.policy, "lab2mj", None)
            if lab2mj is not None and dof_pos.shape[1] == len(lab2mj):
                dof_pos = dof_pos[:, lab2mj]

        if len(dof_pos) == 0:
            return
        if not hasattr(self.policy, "ref_base_pos") or not hasattr(self.policy, "ref_base_quat"):
            return
        curr_idx = min(self.policy.counter_step, len(dof_pos) - 1)
        self._set_freejoint_pose(
            self.ghost_robot_qpos_adr,
            self.ghost_robot_qvel_adr,
            np.concatenate(
                [
                    self.policy.ref_base_pos[curr_idx],
                    self.policy.ref_base_quat[curr_idx],
                ],
                axis=0,
            ),
        )
        q = np.asarray(dof_pos[curr_idx], dtype=np.float32).reshape(-1)
        valid = self.ghost_robot_joint_qpos_adrs >= 0
        self.d.qpos[self.ghost_robot_joint_qpos_adrs[valid]] = q[valid]

    def _reference_object_pose(self) -> np.ndarray | None:
        if not hasattr(self.policy, "ref_object_pos") or not hasattr(self.policy, "ref_object_quat"):
            return None
        if len(self.policy.ref_object_pos) == 0:
            return None
        ref_idx = int(np.clip(int(getattr(self.policy, "counter_step", 1)) - 1, 0, len(self.policy.ref_object_pos) - 1))
        return np.concatenate(
            [
                np.asarray(self.policy.ref_object_pos[ref_idx], dtype=np.float32),
                np.asarray(self.policy.ref_object_quat[ref_idx], dtype=np.float32),
            ],
            axis=-1,
        )

    def _active_ghost_object_adrs(self) -> tuple[int, int]:
        active_name = self._active_object_name()
        if active_name in self.stack_box_names:
            idx = self.stack_box_names.index(active_name)
            return int(self.stack_box_ghost_qpos_adrs[idx]), int(self.stack_box_ghost_qvel_adrs[idx])
        if active_name == "carry_box":
            return self.ghost_carry_box_qpos_adr, self.ghost_carry_box_qvel_adr
        if active_name == "ball" and self.ghost_ball_qpos_adr >= 0:
            return self.ghost_ball_qpos_adr, self.ghost_ball_qvel_adr
        return self.ghost_object_qpos_adr, self.ghost_object_qvel_adr

    def _update_visualization(self):
        ref_object_pose = self._reference_object_pose()
        if ref_object_pose is not None:
            ghost_qpos_adr, ghost_qvel_adr = self._active_ghost_object_adrs()
            self._set_freejoint_pose(
                ghost_qpos_adr,
                ghost_qvel_adr,
                ref_object_pose,
            )

        wrist_goal = self.policy_output.wrist_goal
        if wrist_goal.shape[0] == 14:
            self._set_mocap_pose(self.l_wrist_mocap_id, wrist_goal[:7])
            self._set_mocap_pose(self.r_wrist_mocap_id, wrist_goal[7:14])

        self._set_mocap_pose(self.torso_mocap_id, self.policy_output.torso_goal)
        self._set_mocap_pose(self.l_ankle_mocap_id, self.policy_output.l_ankle_goal)
        self._set_mocap_pose(self.r_ankle_mocap_id, self.policy_output.r_ankle_goal)
        self._update_ghost_robot_visualization()

        contact = self.policy_output.contact
        if contact.shape[0] == 4:
            if self.ref_l_ankle_geom_id >= 0:
                self.m.geom_rgba[self.ref_l_ankle_geom_id] = self.contact_color_on if contact[0] >= 0.5 else self.contact_color_off
            if self.ref_r_ankle_geom_id >= 0:
                self.m.geom_rgba[self.ref_r_ankle_geom_id] = self.contact_color_on if contact[1] >= 0.5 else self.contact_color_off
            if self.ref_l_hand_geom_id >= 0:
                self.m.geom_rgba[self.ref_l_hand_geom_id] = self.contact_color_on if contact[2] >= 0.5 else self.contact_color_off
            if self.ref_r_hand_geom_id >= 0:
                self.m.geom_rgba[self.ref_r_hand_geom_id] = self.contact_color_on if contact[3] >= 0.5 else self.contact_color_off
