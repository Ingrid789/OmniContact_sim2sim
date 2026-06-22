import mujoco
import numpy as np


class OmniContactCarryheartMixin:
    _HEART_SEGMENT_GOAL_PAIRS = (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    )
    _HEART_LOCAL_XY = np.array(
        [
            [-1.360618, 0.012302],
            [-0.792489, 0.789742],
            [-0.003088, 0.436904],
            [0.852096, 0.771801],
            [1.258758, -0.005639],
            [0.947781, -0.687395],
            [0.475337, -1.303366],
            [0.000000, -1.710000],
            [-0.421710, -1.321307],
            [-1.073564, -0.597690],
        ],
        dtype=np.float32,
    )
    _HEART_GOAL_TOLERANCE = 0.2
    _HEART_STATIC_SPEED_THRESHOLD = 0.05
    _HEART_STATIC_ANG_SPEED_THRESHOLD = 0.15
    _HEART_STABLE_TICKS = 10

    def _init_carryheart_state(self) -> None:
        self.carryheart_active_idx = 0
        self.carryheart_all_done = False
        self.carryheart_goal_counter = 0
        self.carryheart_done_flags = []
        self.carryheart_outline_done_flags = []
        self.carryheart_goal_positions = []
        self.carryheart_outline_positions = []
        self.carryheart_execution_to_outline_idx = []
        self.carryheart_init_positions = []
        if not self.is_carryheart:
            return

        half_z = float(self.policy.box_dims[2])
        outline_positions = [
            np.array([xy[0], xy[1], half_z], dtype=np.float32)
            for xy in self._HEART_LOCAL_XY
        ]
        sorted_outline_indices = sorted(
            range(len(outline_positions)),
            key=lambda i: (float(outline_positions[i][1]), abs(float(outline_positions[i][0]))),
        )
        self.carryheart_outline_positions = [p.copy() for p in outline_positions]
        self.carryheart_execution_to_outline_idx = list(sorted_outline_indices)
        self.carryheart_goal_positions = [outline_positions[i].copy() for i in sorted_outline_indices]
        self.carryheart_init_positions = self._sample_carryheart_init_positions()
        self.carryheart_done_flags = [False] * len(self.carryheart_goal_positions)
        self.carryheart_outline_done_flags = [False] * len(self.carryheart_goal_positions)

    def _sample_carryheart_init_positions(self) -> list[np.ndarray]:
        half_z = float(self.policy.box_dims[2])
        count = len(self.carryheart_goal_positions) or len(self._HEART_LOCAL_XY)
        x_min, x_max = -5.0, 5.0
        y_min, y_max = 2.0, 10.0
        x = self.rng.uniform(x_min, x_max, size=count).astype(np.float32)
        y = self.rng.uniform(y_min, y_max, size=count).astype(np.float32)
        xy = np.stack([x, y], axis=1)
        return [np.array([pos[0], pos[1], half_z], dtype=np.float32) for pos in xy]

    def _heart_segment_height(self) -> float:
        return float(self.policy.box_dims[2]) + 0.01

    def _heart_segment_quat(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        delta = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
        yaw = float(np.arctan2(delta[1], delta[0]))
        half = 0.5 * yaw
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    def _update_heart_segment_visuals(self):
        if not self.is_carryheart:
            return
        pending_rgba = np.array([0.70, 0.70, 0.70, 0.18], dtype=np.float32)
        done_rgba = np.array([0.933, 0.388, 0.388, 0.90], dtype=np.float32)
        z = self._heart_segment_height()

        for (start_idx, end_idx), mocap_id, geom_id in zip(
            self._HEART_SEGMENT_GOAL_PAIRS,
            self.heart_segment_mocap_ids,
            self.heart_segment_geom_ids,
        ):
            if mocap_id < 0 or geom_id < 0:
                continue
            start = self.carryheart_outline_positions[start_idx].copy()
            end = self.carryheart_outline_positions[end_idx].copy()
            start[2] = z
            end[2] = z
            center = 0.5 * (start + end)
            length = float(np.linalg.norm(end[:2] - start[:2]))
            self.d.mocap_pos[mocap_id] = center
            self.d.mocap_quat[mocap_id] = self._heart_segment_quat(start, end)
            self.m.geom_size[geom_id] = np.array([0.5 * length, 0.035, 0.01], dtype=np.float32)
            segment_done = self.carryheart_outline_done_flags[start_idx] and self.carryheart_outline_done_flags[end_idx]
            self.m.geom_rgba[geom_id] = done_rgba if segment_done else pending_rgba

    def _set_active_policy_goal(self):
        if self.is_carryheart:
            self.policy.goal_pos_override = self._goal_pos()
            if self.plane2_mocap_id >= 0:
                self.d.mocap_pos[self.plane2_mocap_id] = self._plane2_goal_pos() - np.array(
                    [0.0, 0.0, self._table_height_offset() + 0.01],
                    dtype=np.float32,
                )

    def _set_carryheart_box_pose(self, idx: int, pos: np.ndarray):
        if idx < 0 or idx >= len(self.carryheart_box_qpos_adrs):
            return
        qpos_adr = int(self.carryheart_box_qpos_adrs[idx])
        qvel_adr = int(self.carryheart_box_qvel_adrs[idx])
        if qpos_adr >= 0:
            self.d.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float32).reshape(3)
            self.d.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if qvel_adr >= 0:
            self.d.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _reset_carryheart_env(self):
        self.carryheart_init_positions = self._sample_carryheart_init_positions()
        if len(self.carryheart_box_qpos_adrs) < len(self.carryheart_init_positions):
            raise ValueError("carryheart requires 10 box freejoints in the Mujoco XML.")
        for i, pos in enumerate(self.carryheart_init_positions):
            self._set_carryheart_box_pose(i, pos)

        self.carryheart_active_idx = 0
        self.carryheart_all_done = False
        self.carryheart_goal_counter = 0
        self.carryheart_done_flags = [False] * len(self.carryheart_goal_positions)
        self.carryheart_outline_done_flags = [False] * len(self.carryheart_goal_positions)
        self.policy.active_object_name = self._active_object_name()
        self._set_active_policy_goal()
        self._update_heart_segment_visuals()

        active_body_id = self._get_active_object_body_id()
        if self.plane1_mocap_id >= 0 and active_body_id >= 0:
            self.d.mocap_pos[self.plane1_mocap_id] = self.d.xpos[active_body_id].copy() - np.array(
                [0.0, 0.0, self._table_height_offset() + 0.01],
                dtype=np.float32,
            )
        mujoco.mj_step(self.m, self.d)

    def _carryheart_box_speed(self, idx: int) -> tuple[float, float]:
        if idx < 0 or idx >= len(self.carryheart_box_qvel_adrs):
            return 0.0, 0.0
        qvel_adr = int(self.carryheart_box_qvel_adrs[idx])
        if qvel_adr < 0 or qvel_adr + 6 > self.d.qvel.shape[0]:
            return 0.0, 0.0
        lin_speed = float(np.linalg.norm(self.d.qvel[qvel_adr : qvel_adr + 3]))
        ang_speed = float(np.linalg.norm(self.d.qvel[qvel_adr + 3 : qvel_adr + 6]))
        return lin_speed, ang_speed

    def _active_object_near_goal(self) -> bool:
        active_body_id = self._get_active_object_body_id()
        if active_body_id < 0:
            return False
        return float(np.linalg.norm(self.d.xpos[active_body_id] - self._goal_pos())) <= self._HEART_GOAL_TOLERANCE

    def _is_active_object_held(self) -> bool:
        active_body_id = self._get_active_object_body_id()
        if active_body_id < 0:
            return False
        obj_pos = self.d.xpos[active_body_id]
        dists = []
        if self.left_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - self.d.xpos[self.left_palm_body_id])))
        if self.right_palm_body_id >= 0:
            dists.append(float(np.linalg.norm(obj_pos - self.d.xpos[self.right_palm_body_id])))
        return bool(dists) and min(dists) <= 0.32

    def _restart_policy_from_current_state(self):
        self.policy.replan_active = False
        self.replan.reset_detection()
        self._set_active_policy_goal()
        self._sync_state_cmd_from_mj()
        self.policy.enter()
        self._sync_state_cmd_from_mj()
        self.policy_output.switch_to_loco = False
        self.policy_output.success = ""

    def _set_carryheart_active_box(self, idx: int) -> None:
        self.carryheart_active_idx = int(idx)
        self.carryheart_goal_counter = 0
        self.carryheart_all_done = False
        self.policy.active_object_name = self._active_object_name()
        self._update_heart_segment_visuals()

        active_body_id = self._get_active_object_body_id()
        if self.plane1_mocap_id >= 0 and active_body_id >= 0:
            self.d.mocap_pos[self.plane1_mocap_id] = self.d.xpos[active_body_id].copy() - np.array(
                [0.0, 0.0, self._table_height_offset() + 0.01],
                dtype=np.float32,
            )
        self._restart_policy_from_current_state()

    def _monitor_carryheart_done_boxes(self) -> bool:
        if not self.is_carryheart or self.carryheart_all_done:
            return False

        for idx, done in enumerate(self.carryheart_done_flags):
            if not done or idx == self.carryheart_active_idx:
                continue
            body_id = int(self.carryheart_box_body_ids[idx])
            if body_id < 0:
                continue
            goal = self.carryheart_goal_positions[idx]
            dist = float(np.linalg.norm(self.d.xpos[body_id] - goal))
            if dist <= self._HEART_GOAL_TOLERANCE:
                continue
            lin_speed, ang_speed = self._carryheart_box_speed(idx)
            is_static = lin_speed <= self._HEART_STATIC_SPEED_THRESHOLD and ang_speed <= self._HEART_STATIC_ANG_SPEED_THRESHOLD
            if not is_static:
                continue

            self.carryheart_done_flags[idx] = False
            outline_idx = int(self.carryheart_execution_to_outline_idx[idx])
            self.carryheart_outline_done_flags[outline_idx] = False
            print(
                f"[heart] box#{idx + 1} drifted from goal by {dist:.3f} m; "
                f"replan to recover it."
            )
            self._set_carryheart_active_box(idx)
            return True
        return False

    def _maybe_advance_carryheart(self):
        if not self.is_carryheart or self.carryheart_all_done:
            return

        if not bool(getattr(self.policy_output, "switch_to_loco", False)):
            self.carryheart_goal_counter = 0
            return

        near_goal = self._active_object_near_goal()
        held = self._is_active_object_held()
        lin_speed, ang_speed = self._active_object_speed()
        is_static = lin_speed <= self._HEART_STATIC_SPEED_THRESHOLD and ang_speed <= self._HEART_STATIC_ANG_SPEED_THRESHOLD

        if near_goal and (not held) and is_static:
            self.carryheart_goal_counter += 1
        else:
            self.carryheart_goal_counter = 0

        if self.carryheart_goal_counter < self._HEART_STABLE_TICKS:
            return

        done_idx = int(self.carryheart_active_idx)
        self.carryheart_done_flags[done_idx] = True
        outline_idx = int(self.carryheart_execution_to_outline_idx[done_idx])
        self.carryheart_outline_done_flags[outline_idx] = True
        print(
            f"[heart] box#{done_idx + 1} placed at goal {self.carryheart_goal_positions[done_idx].tolist()} "
            f"| lin={lin_speed:.3f} m/s, ang={ang_speed:.3f} rad/s"
        )
        self.carryheart_goal_counter = 0

        next_idx = done_idx + 1

        if next_idx >= len(self.carryheart_goal_positions):
            self.carryheart_all_done = True
            self._update_heart_segment_visuals()
            print("[heart] all 10 boxes placed into heart shape. finish.")
            return

        self._set_carryheart_active_box(next_idx)
