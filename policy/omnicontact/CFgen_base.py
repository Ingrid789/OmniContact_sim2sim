import numpy as np

from common.utils import (
    align_quat_hemisphere,
    normalize_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_slerp,
    yaw_quat,
    yaw_to_quat,
)
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_JOINT_POS_MJ,
    DEFAULT_PELVIS_Z,
    KINEMATICS,
    _append_loco_approach,
    _quat_to_rpy_deg,
)
from policy.omnicontact.CFgen_builder import _TrajBuilder


class CfGenBase:
    """Shared utilities for rule-based OmniContact reference generators."""

    def __init__(self) -> None:
        # contact flags: [left_foot, right_foot, left_hand, right_hand]
        self._contact0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._contact_lfoot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._contact_rfoot = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._contact_hand = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        self._contact1_hands = self._contact_hand

        self._left_palm_center = np.array([0.17, 0.015, 0.0], dtype=np.float32)
        self._right_palm_center = np.array([0.17, -0.015, 0.0], dtype=np.float32)
        self._left_palm_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        self._right_palm_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._vec_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.ik_params = {
            "ik_palm_pos_weight": 20.0,
            "ik_palm_faceobj_weight": 4.0,
        }

    def _new_builder(self) -> _TrajBuilder:
        return _TrajBuilder()

    @staticmethod
    def _quat_slerp_sequence(q_start: np.ndarray, q_end: np.ndarray, n: int) -> np.ndarray:
        u = np.linspace(0.0, 1.0, max(int(n), 2), dtype=np.float32)
        return align_quat_hemisphere(
            np.array([normalize_quat(quat_slerp(q_start, q_end, float(t))) for t in u], dtype=np.float32)
        )

    @staticmethod
    def _joint_lerp(q_start: np.ndarray, q_end: np.ndarray, n: int) -> np.ndarray:
        u = np.linspace(0.0, 1.0, max(int(n), 2), dtype=np.float32)
        q_start = np.asarray(q_start, dtype=np.float32).reshape(1, -1)
        q_end = np.asarray(q_end, dtype=np.float32).reshape(1, -1)
        return ((1.0 - u[:, None]) * q_start + u[:, None] * q_end).astype(np.float32)

    @staticmethod
    def _vec_lerp(v_start: np.ndarray, v_end: np.ndarray, n: int) -> np.ndarray:
        u = np.linspace(0.0, 1.0, max(int(n), 2), dtype=np.float32)
        v_start = np.asarray(v_start, dtype=np.float32).reshape(1, -1)
        v_end = np.asarray(v_end, dtype=np.float32).reshape(1, -1)
        return ((1.0 - u[:, None]) * v_start + u[:, None] * v_end).astype(np.float32)

    def _append_loco_approach_with_waypoints(
        self,
        b: _TrajBuilder,
        *,
        phase_turn_to_walk: int = 11,
        phase_walk: int = 12,
        phase_turn_to_target: int = 13,
        pelvis_start: np.ndarray,
        pelvis_target: np.ndarray,
        yaw_start: np.ndarray,
        yaw_target: np.ndarray,
        step_linear: float,
        step_angular: float,
        object_pos: np.ndarray,
        object_quat: np.ndarray,
        obstacle_half_dims: np.ndarray,
        waypoint_trigger_margin: float | None = None,
        waypoint_trigger_distance: float | None = None,
        obstacle_margin: float | None = None,
        waypoint_clearance: float | None = None,
    ) -> None:
        cfg = getattr(self, "cfg", {})
        start = np.asarray(pelvis_start, dtype=np.float32).reshape(3)
        target = np.asarray(pelvis_target, dtype=np.float32).reshape(3)
        obstacle = np.asarray(object_pos, dtype=np.float32).reshape(3)
        dims = np.asarray(obstacle_half_dims, dtype=np.float32).reshape(3)

        trigger_margin = (
            float(cfg.get("phase11_waypoint_trigger_margin", 0.05))
            if waypoint_trigger_margin is None
            else float(waypoint_trigger_margin)
        )
        trigger_distance = (
            float(cfg.get("phase11_waypoint_trigger_distance", 0.2))
            if waypoint_trigger_distance is None
            else float(waypoint_trigger_distance)
        )
        margin = (
            float(cfg.get("phase11_obstacle_margin", 0.55))
            if obstacle_margin is None
            else float(obstacle_margin)
        )
        clearance = (
            float(cfg.get("phase11_waypoint_clearance", 0.75))
            if waypoint_clearance is None
            else float(waypoint_clearance)
        )

        trigger_half = np.maximum(dims[:2] + trigger_margin, trigger_margin)
        box_min = obstacle[:2] - trigger_half
        box_max = obstacle[:2] + trigger_half

        segment = target[:2] - start[:2]
        t_min = 0.0
        t_max = 1.0
        intersects_trigger_box = True
        for axis in range(2):
            if abs(float(segment[axis])) < 1e-6:
                if start[axis] < box_min[axis] or start[axis] > box_max[axis]:
                    intersects_trigger_box = False
                    break
                continue

            inv_d = 1.0 / float(segment[axis])
            t1 = float((box_min[axis] - start[axis]) * inv_d)
            t2 = float((box_max[axis] - start[axis]) * inv_d)
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            if t_min > t_max:
                intersects_trigger_box = False
                break

        segment_len_sq = float(np.dot(segment, segment))
        if segment_len_sq < 1e-8:
            center_path_dist = float(np.linalg.norm(obstacle[:2] - start[:2]))
        else:
            t = float(np.clip(np.dot(obstacle[:2] - start[:2], segment) / segment_len_sq, 0.0, 1.0))
            center_path_dist = float(np.linalg.norm(obstacle[:2] - (start[:2] + t * segment)))

        waypoints = [target.copy()]
        corridor_radius = float(np.max(dims[:2]) + trigger_distance)
        if intersects_trigger_box or center_path_dist <= corridor_radius:
            path_dir = target[:2] - start[:2]
            norm = float(np.linalg.norm(path_dir))
            if norm >= 1e-6:
                path_dir /= norm
                side_dir = np.array([-path_dir[1], path_dir[0]], dtype=np.float32)
                route_half = np.maximum(dims[:2] + margin, margin)

                candidates = []
                for sign in (1.0, -1.0):
                    side = side_dir * sign
                    support = route_half[0] * abs(float(side[0])) + route_half[1] * abs(float(side[1]))
                    waypoint = obstacle.copy()
                    waypoint[:2] = obstacle[:2] + side * (support + clearance)
                    waypoint[2] = DEFAULT_PELVIS_Z
                    path_len = float(
                        np.linalg.norm(waypoint[:2] - start[:2])
                        + np.linalg.norm(target[:2] - waypoint[:2])
                    )
                    candidates.append((path_len, waypoint.astype(np.float32)))
                waypoints = [min(candidates, key=lambda item: item[0])[1], target.copy()]

        current_pos = start.copy()
        current_yaw = np.asarray(yaw_start, dtype=np.float32).reshape(4).copy()
        for idx, waypoint in enumerate(waypoints):
            if idx == len(waypoints) - 1:
                segment_yaw_target = yaw_target
            else:
                next_delta = waypoints[idx + 1][:2] - waypoint[:2]
                if float(np.linalg.norm(next_delta)) < 1e-6:
                    segment_yaw_target = current_yaw
                else:
                    segment_yaw_target = yaw_to_quat(np.arctan2(float(next_delta[1]), float(next_delta[0]))).astype(
                        np.float32
                    )

            _append_loco_approach(
                b,
                phase_turn_to_walk=phase_turn_to_walk,
                phase_walk=phase_walk,
                phase_turn_to_target=phase_turn_to_target,
                pelvis_start=current_pos,
                pelvis_target=waypoint,
                yaw_start=current_yaw,
                yaw_target=segment_yaw_target,
                step_linear=step_linear,
                step_angular=step_angular,
                object_pos=object_pos,
                object_quat=object_quat,
            )
            current_pos = b.last("base_p").copy()
            current_yaw = b.last("base_q").copy()

    def _solve_fixed_feet_hand_ik(
        self,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        left_ankle_pos: np.ndarray,
        left_ankle_quat: np.ndarray,
        right_ankle_pos: np.ndarray,
        right_ankle_quat: np.ndarray,
        left_palm_pos: np.ndarray,
        left_palm_quat: np.ndarray | None = None,
        right_palm_pos: np.ndarray,
        right_palm_quat: np.ndarray | None = None,
        seed_joint_pos: np.ndarray,
        pelvis_z_min: float = 0.0,
        ik_palm_pos_weight: float = 0.0,
        ik_palm_faceobj_weight: float = 0.0,
        object_pos: np.ndarray | None = None,
        object_quat: np.ndarray | None = None,
        torso_pos: np.ndarray | None = None,
        torso_quat: np.ndarray | None = None,
        ik_joint_indices: np.ndarray | None = None,
        waist_pitch_min_rad: float = 0.0,
        ankle_pos_weight: float = 10.0,
        ankle_quat_weight: float = 4.0,
        torso_pos_weight: float = 20.0,
        torso_quat_weight: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.optimize import least_squares

        def quat_error_vec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
            q_err = normalize_quat(quat_mul(normalize_quat(target), quat_conjugate(normalize_quat(current))))
            sign = 1.0 if float(q_err[0]) >= 0.0 else -1.0
            return (2.0 * sign * q_err[1:]).astype(np.float32)

        def vec3(value: np.ndarray) -> np.ndarray:
            return np.asarray(value, dtype=np.float64).reshape(3)

        def optional_vec3(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else vec3(value)

        def optional_quat(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else normalize_quat(value)

        def unit(v: np.ndarray) -> np.ndarray:
            return v / (np.linalg.norm(v) + 1e-8)

        def pitch_delta_quat(pitch_rad: float) -> np.ndarray:
            half = 0.5 * float(pitch_rad)
            return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float32)

        active = (
            np.asarray(ik_joint_indices, dtype=np.int32).reshape(-1)
            if ik_joint_indices is not None
            else np.asarray(self.cfg["ik_joint_indices"], dtype=np.int32).reshape(-1)
        )
        joint_lower = np.zeros(len(active), dtype=np.float32)
        joint_upper = np.zeros(len(active), dtype=np.float32)
        for i, joint_idx in enumerate(active):
            joint_range = KINEMATICS.model.joint(KINEMATICS.joint_names[int(joint_idx)]).range
            joint_lower[i] = float(joint_range[0])
            joint_upper[i] = float(joint_range[1])

        q_seed = np.asarray(seed_joint_pos, dtype=np.float64).reshape(-1).copy()
        pelvis_seed = vec3(pelvis_pos).copy()
        pelvis_quat = normalize_quat(pelvis_quat)
        left_ankle_pos = vec3(left_ankle_pos)
        left_ankle_quat = normalize_quat(left_ankle_quat)
        right_ankle_pos = vec3(right_ankle_pos)
        right_ankle_quat = normalize_quat(right_ankle_quat)
        left_palm_pos = vec3(left_palm_pos)
        right_palm_pos = vec3(right_palm_pos)
        left_palm_quat = optional_quat(left_palm_quat)
        right_palm_quat = optional_quat(right_palm_quat)
        object_pos = optional_vec3(object_pos)
        torso_pos = optional_vec3(torso_pos)
        torso_quat = optional_quat(torso_quat)

        pelvis_pos_dim = 1
        pelvis_pitch_dim = 1
        joint_offset = pelvis_pos_dim + pelvis_pitch_dim
        x_seed = np.concatenate(
            [
                pelvis_seed[2:3].copy(),
                np.zeros(1, dtype=np.float64),
                np.clip(q_seed[active], joint_lower, joint_upper),
            ]
        )
        lower = np.concatenate(
            [
                np.array([min(float(pelvis_z_min), float(pelvis_seed[2]) - 1e-3)], dtype=np.float64),
                np.array([0.0], dtype=np.float64),
                joint_lower,
            ]
        )
        upper = np.concatenate(
            [
                np.array([DEFAULT_PELVIS_Z], dtype=np.float64),
                np.array([np.deg2rad(60.0)], dtype=np.float64),
                joint_upper,
            ]
        )
        waist_pitch_active_pos = np.flatnonzero(active == 14)
        if waist_pitch_active_pos.size:
            idx = joint_offset + int(waist_pitch_active_pos[0])
            lower[idx] = max(lower[idx], float(waist_pitch_min_rad))

        def residual(x: np.ndarray) -> np.ndarray:
            pelvis = pelvis_seed.copy()
            pelvis[2] = x[0]
            pelvis_quat_i = normalize_quat(quat_mul(pelvis_quat, pitch_delta_quat(float(x[pelvis_pos_dim]))))
            q = q_seed.copy()
            q[active] = x[joint_offset:]
            fk = KINEMATICS.forward(q, pelvis, pelvis_quat_i)

            l_wp = fk["left_wrist_yaw_link"]["pos"]
            l_wq = fk["left_wrist_yaw_link"]["quat"]
            r_wp = fk["right_wrist_yaw_link"]["pos"]
            r_wq = fk["right_wrist_yaw_link"]["quat"]
            l_palm_pos_tmp = l_wp + quat_apply(l_wq, self._left_palm_center)
            r_palm_pos_tmp = r_wp + quat_apply(r_wq, self._right_palm_center)

            residuals = []
            if torso_pos is not None:
                residuals.append(float(torso_pos_weight) * (fk["torso_link"]["pos"] - torso_pos))
            if torso_quat is not None:
                residuals.append(
                    float(torso_quat_weight)
                    * quat_error_vec(torso_quat, fk["torso_link"]["quat"].astype(np.float32))
                )

            residuals.extend(
                [
                    float(ankle_pos_weight) * (fk["left_ankle_pitch_link"]["pos"] - left_ankle_pos),
                    float(ankle_pos_weight) * (fk["right_ankle_pitch_link"]["pos"] - right_ankle_pos),
                    float(ankle_quat_weight)
                    * quat_error_vec(left_ankle_quat, fk["left_ankle_pitch_link"]["quat"].astype(np.float32)),
                    float(ankle_quat_weight)
                    * quat_error_vec(right_ankle_quat, fk["right_ankle_pitch_link"]["quat"].astype(np.float32)),
                    ik_palm_pos_weight * (l_palm_pos_tmp - left_palm_pos),
                    ik_palm_pos_weight * (r_palm_pos_tmp - right_palm_pos),
                ]
            )

            has_palm_quat = left_palm_quat is not None or right_palm_quat is not None
            if has_palm_quat:
                if left_palm_quat is not None:
                    residuals.append(2.0 * quat_error_vec(left_palm_quat, l_wq))
                if right_palm_quat is not None:
                    residuals.append(2.0 * quat_error_vec(right_palm_quat, r_wq))
            elif object_pos is not None:
                residuals.append(
                    ik_palm_faceobj_weight
                    * (quat_apply(l_wq, self._left_palm_normal) - unit(object_pos - l_palm_pos_tmp))
                )
                residuals.append(
                    ik_palm_faceobj_weight
                    * (quat_apply(r_wq, self._right_palm_normal) - unit(object_pos - r_palm_pos_tmp))
                )

            residuals.extend(
                [
                    0.1 * (x[:joint_offset] - x_seed[:joint_offset]),
                    0.03 * (x[joint_offset:] - x_seed[joint_offset:]),
                ]
            )
            return np.concatenate(residuals)

        result = least_squares(
            residual,
            x_seed.astype(np.float64),
            bounds=(lower.astype(np.float64), upper.astype(np.float64)),
            max_nfev=100,
            diff_step=1e-4,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4,
        )
        pelvis_out = pelvis_seed.copy()
        pelvis_out[2] = result.x[0]
        pelvis_quat_out = normalize_quat(quat_mul(pelvis_quat, pitch_delta_quat(float(result.x[pelvis_pos_dim]))))
        q_out = q_seed.copy()
        q_out[active] = result.x[joint_offset:]
        return pelvis_out.astype(np.float32), pelvis_quat_out.astype(np.float32), q_out.astype(np.float32)

    def _append_joint_fk_phase(
        self,
        b: _TrajBuilder,
        phase: int,
        *,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        qseq: np.ndarray,
        pelvis_seq: np.ndarray,
        pelvis_quat_seq: np.ndarray,
        contact: np.ndarray,
    ) -> None:
        qseq = np.asarray(qseq, dtype=np.float32).reshape(-1, len(DEFAULT_JOINT_POS_MJ))
        n = int(len(qseq))

        obj_p = np.asarray(obj_pos, dtype=np.float32)
        obj_q = np.asarray(obj_quat, dtype=np.float32)
        if obj_p.ndim == 1:
            obj_p = np.tile(obj_p.reshape(1, 3), (n, 1))
        if obj_q.ndim == 1:
            obj_q = np.tile(obj_q.reshape(1, 4), (n, 1))

        pelvis_seq = np.asarray(pelvis_seq, dtype=np.float32)
        if pelvis_seq.ndim == 1:
            pelvis_seq = np.repeat(pelvis_seq.reshape(1, 3), n, axis=0)
        pelvis_seq = pelvis_seq.reshape(n, 3)

        pelvis_quat_seq = np.asarray(pelvis_quat_seq, dtype=np.float32)
        if pelvis_quat_seq.ndim == 1:
            pelvis_quat_seq = np.repeat(pelvis_quat_seq.reshape(1, 4), n, axis=0)
        pelvis_quat_seq = align_quat_hemisphere(pelvis_quat_seq.reshape(n, 4))

        refs = {
            "lw_p": np.zeros((n, 3), dtype=np.float32),
            "lw_q": np.zeros((n, 4), dtype=np.float32),
            "rw_p": np.zeros((n, 3), dtype=np.float32),
            "rw_q": np.zeros((n, 4), dtype=np.float32),
            "torso_p": np.zeros((n, 3), dtype=np.float32),
            "torso_yaw_q": np.zeros((n, 4), dtype=np.float32),
            "torso_pitch_deg": np.zeros(n, dtype=np.float32),
            "la_p": np.zeros((n, 3), dtype=np.float32),
            "la_q": np.zeros((n, 4), dtype=np.float32),
            "ra_p": np.zeros((n, 3), dtype=np.float32),
            "ra_q": np.zeros((n, 4), dtype=np.float32),
        }
        for i, q in enumerate(qseq):
            fk = KINEMATICS.forward(q, pelvis_seq[i], pelvis_quat_seq[i])
            torso_quat = normalize_quat(fk["torso_link"]["quat"])
            torso_yaw = normalize_quat(yaw_quat(torso_quat))
            torso_rel = normalize_quat(quat_mul(quat_conjugate(torso_yaw), torso_quat))
            _, pitch_deg, _ = _quat_to_rpy_deg(torso_rel)

            refs["lw_p"][i] = fk["left_palm_link"]["pos"]
            refs["lw_q"][i] = fk["left_palm_link"]["quat"]
            refs["rw_p"][i] = fk["right_palm_link"]["pos"]
            refs["rw_q"][i] = fk["right_palm_link"]["quat"]
            refs["torso_p"][i] = fk["torso_link"]["pos"]
            refs["torso_yaw_q"][i] = torso_yaw
            refs["torso_pitch_deg"][i] = float(pitch_deg)
            refs["la_p"][i] = fk["left_ankle_pitch_link"]["pos"]
            refs["la_q"][i] = fk["left_ankle_pitch_link"]["quat"]
            refs["ra_p"][i] = fk["right_ankle_pitch_link"]["pos"]
            refs["ra_q"][i] = fk["right_ankle_pitch_link"]["quat"]

        for key in ("lw_q", "rw_q", "torso_yaw_q", "la_q", "ra_q"):
            refs[key] = align_quat_hemisphere(refs[key])

        b.append(
            int(phase),
            lw_p=refs["lw_p"],
            lw_q=refs["lw_q"],
            rw_p=refs["rw_p"],
            rw_q=refs["rw_q"],
            obj_p=obj_p,
            obj_q=obj_q,
            torso_p=refs["torso_p"],
            torso_yaw_q=refs["torso_yaw_q"],
            torso_pitch_deg=refs["torso_pitch_deg"],
            la_p=refs["la_p"],
            la_q=refs["la_q"],
            ra_p=refs["ra_p"],
            ra_q=refs["ra_q"],
            dof_pos=qseq,
            base_p=pelvis_seq,
            base_q=pelvis_quat_seq,
            contact=contact,
        )
