import numpy as np

from common.utils import (
    normalize_quat,
    yaw_quat,
    yaw_to_quat,
)
from policy.omnicontact.CFgen_base import CfGenBase
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_PELVIS_Z,
    KINEMATICS,
    _append_contactloco_recover,
    _append_fk_block,
    _append_loco_walk,
    _fk_reference_sequence_from_joints,
    _lerp_sequence,
)


class CfGenKickBall(CfGenBase):
    """Standalone kick-ball planner.

    This is a direct slidebox-style synthesis adapted for a ground ball target:
      11. Turn and walk to an intermediate stance 1.0 m behind the ball.
      12. Walk forward to the final kick stance 0.2 m behind the ball.
      13. Lift the kicking leg backward with FK.
      14. Swing the kicking leg forward with FK, using foot contact in the second half.
      15. Recover back to the default standing pose.
      16. Hold the terminal standing pose.
    """

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__()
        self.pad = int(pad)
        self.step_linear = float(step_size_linear)
        self.step_angular = float(step_size_angular)

        self.cfg = {
            "phase11_standoff_dist": 1.5,
            "phase12_standoff_dist": 0.2,
            "phase11_waypoint_trigger_margin": 0.03,
            "phase11_waypoint_trigger_distance": 0.12,
            "phase11_obstacle_margin": 0.35,
            "phase11_waypoint_clearance": 0.45,
            "kick_lateral_offset": 0.12,
            "phase13_lift_frames": 2,
            "phase14_swing_frames": 10,
            "phase15_recover_frames": 10,
            "phase13_hip_swing_deg": 30.0,
            "phase13_knee_swing_deg": 0.0,
            "phase14_hip_swing_deg": -60.0,
            "phase14_knee_swing_deg": 0.0,
        }

        joint_names = KINEMATICS.joint_names
        self._right_hip_pitch_idx = joint_names.index("right_hip_pitch_joint")
        self._right_knee_idx = joint_names.index("right_knee_joint")

    @staticmethod
    def _move_direction(
        obj_pos: np.ndarray,
        torso_pos: np.ndarray,
        target_obj_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        move_dir = (target_obj_pos[:2] - obj_pos[:2]).astype(np.float32)
        move_norm = float(np.linalg.norm(move_dir))
        if move_norm < 1e-6:
            move_dir = (obj_pos[:2] - torso_pos[:2]).astype(np.float32)
            move_norm = float(np.linalg.norm(move_dir))
        if move_norm < 1e-6:
            move_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            move_dir = (move_dir / move_norm).astype(np.float32)
        target_yaw = float(np.arctan2(float(move_dir[1]), float(move_dir[0])))
        target_yaw_quat = yaw_to_quat(target_yaw).astype(np.float32)
        return move_dir.astype(np.float32), target_yaw_quat, target_yaw

    def generate(
        self,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        box_half_dims: np.ndarray = np.array([0.15, 0.15, 0.15]),
        target_obj_pos: np.ndarray = np.array([1.0, 1.0, 0.15]),
        task: str = "kickball",
    ):
        pelvis_pos0 = np.asarray(pelvis_pos, dtype=np.float32).reshape(3).copy()
        pelvis_quat0 = normalize_quat(pelvis_quat)
        obj_pos0 = np.asarray(obj_pos, dtype=np.float32).reshape(3).copy()
        obj_quat0 = normalize_quat(obj_quat)
        target_obj_pos = np.asarray(target_obj_pos, dtype=np.float32).reshape(3).copy()
        box_half_dims = np.asarray(box_half_dims, dtype=np.float32).reshape(3)
        target_obj_pos[2] = float(box_half_dims[2])

        move_dir, target_yaw_quat, target_yaw = self._move_direction(
            obj_pos0,
            pelvis_pos0,
            target_obj_pos,
        )

        b = self._new_builder()
        move_left_dir = np.array([-move_dir[1], move_dir[0]], dtype=np.float32)
        stance_lateral_offset = (move_left_dir * float(self.cfg["kick_lateral_offset"])).astype(np.float32)
        phase11_stance_pos = np.array(
            [
                obj_pos0[0] - move_dir[0] * float(self.cfg["phase11_standoff_dist"]),
                obj_pos0[1] - move_dir[1] * float(self.cfg["phase11_standoff_dist"]),
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )
        phase11_stance_pos[:2] += stance_lateral_offset
        phase12_stance_pos = np.array(
            [
                obj_pos0[0] - move_dir[0] * float(self.cfg["phase12_standoff_dist"]),
                obj_pos0[1] - move_dir[1] * float(self.cfg["phase12_standoff_dist"]),
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )
        phase12_stance_pos[:2] += stance_lateral_offset

        # -------------------------
        # Phase11: turn and walk to an intermediate stance 1.0 m behind the ball.
        # -------------------------
        self._append_loco_approach_with_waypoints(
            b,
            phase_turn_to_walk=11,
            phase_walk=11,
            phase_turn_to_target=11,
            pelvis_start=pelvis_pos0,
            pelvis_target=phase11_stance_pos,
            yaw_start=yaw_quat(pelvis_quat0),
            yaw_target=target_yaw_quat,
            step_linear=self.step_linear,
            step_angular=self.step_angular,
            object_pos=obj_pos0,
            object_quat=obj_quat0,
            obstacle_half_dims=box_half_dims,
        )

        # -------------------------
        # Phase12: walk forward to the final kick stance 0.2 m behind the ball.
        # -------------------------
        _append_loco_walk(
            b,
            12,
            pelvis_start=b.last("base_p"),
            pelvis_target=phase12_stance_pos,
            yaw=target_yaw_quat,
            step_linear=self.step_linear*1.5,
            object_pos=b.last("obj_p"),
            object_quat=b.last("obj_q"),
        )

        # -------------------------
        # Phase13: FK-based backswing lift for the kicking leg.
        # -------------------------
        n13 = int(max(int(self.cfg["phase13_lift_frames"]), 2))
        dof13_start = b.last("dof_pos").copy().astype(np.float32)
        dof13_target = dof13_start.copy()
        dof13_target[self._right_hip_pitch_idx] = np.deg2rad(float(self.cfg["phase13_hip_swing_deg"]))
        dof13_target[self._right_knee_idx] = np.deg2rad(float(self.cfg["phase13_knee_swing_deg"]))
        dof13 = _lerp_sequence(dof13_start, dof13_target, n13)
        pelvis13 = np.tile(b.last("base_p").reshape(1, 3), (n13, 1)).astype(np.float32)
        base_q13 = np.tile(b.last("base_q").reshape(1, 4), (n13, 1)).astype(np.float32)
        refs13 = _fk_reference_sequence_from_joints(pelvis13, base_q13, dof13)
        _append_fk_block(
            b,
            13,
            fk_refs=refs13,
            object_pos=b.last("obj_p"),
            object_quat=b.last("obj_q"),
            contact=np.tile(self._contact0.reshape(1, 4), (n13, 1)).astype(np.float32),
        )

        # -------------------------
        # Phase14: FK-based forward swing, with kicking-foot contact on the final frames.
        # -------------------------
        n14 = int(max(int(self.cfg["phase14_swing_frames"]), 2))
        dof14_start = b.last("dof_pos").copy().astype(np.float32)
        dof14_target = dof14_start.copy()
        dof14_target[self._right_hip_pitch_idx] = np.deg2rad(float(self.cfg["phase14_hip_swing_deg"]))
        dof14_target[self._right_knee_idx] = np.deg2rad(float(self.cfg["phase14_knee_swing_deg"]))
        dof14 = _lerp_sequence(dof14_start, dof14_target, n14)
        pelvis14 = np.tile(b.last("base_p").reshape(1, 3), (n14, 1)).astype(np.float32)
        base_q14 = np.tile(b.last("base_q").reshape(1, 4), (n14, 1)).astype(np.float32)
        refs14 = _fk_reference_sequence_from_joints(pelvis14, base_q14, dof14)
        contact14 = np.tile(self._contact0.reshape(1, 4), (n14, 1)).astype(np.float32)
        contact14[n14//2:] = self._contact_rfoot
        _append_fk_block(
            b,
            14,
            fk_refs=refs14,
            object_pos=b.last("obj_p"),
            object_quat=b.last("obj_q"),
            contact=contact14,
        )

        # -------------------------
        # Phase15: recover back to the default standing pose.
        # -------------------------
        _append_contactloco_recover(
            b,
            15,
            recover_frames=int(self.cfg["phase15_recover_frames"]),
            recover_contact=self._contact0,
            recover_pelvis_z=DEFAULT_PELVIS_Z,
            recover_pelvis_pitch=True,
        )

        # -------------------------
        # Phase16: hold the terminal standing pose.
        # -------------------------
        b.pad(16, contact=self._contact0, count=self.pad)
        return b.finalize(), target_yaw
