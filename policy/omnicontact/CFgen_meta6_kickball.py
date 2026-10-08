"""Single-shot kick references for the G1 OmniContact controller.

The policy tracks Cartesian body targets. Construct a reachable foot path with
leg IK instead of a two-frame hip-angle jump, and keep the left foot planted
through the swing. Parameters were selected using separate development cases;
the actual goal and the downstream success rule are never modified here.
"""
import numpy as np

from common.utils import normalize_quat, yaw_to_quat
from policy.omnicontact.CFgen_base import CfGenBase
from policy.omnicontact.CFgen_kickball_cartesian import generate_cartesian


class CfGenKickBall(CfGenBase):
    """Approach, settle, load the support leg, strike once, recover and hold.

    All frame counts use the existing 50 Hz reference clock. A 15 degree aim
    correction rotates the reference only, not the scene goal. The stance,
    backswing height and follow-through were calibrated for the 42k RSL
    transformer. These are empirical tracking parameters, not a guarantee
    for other checkpoints or ball geometries.
    """

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__()
        self.pad = int(pad)
        self.step_linear = float(step_size_linear)
        self.step_angular = float(step_size_angular)
        self.cfg = {
            "phase11_waypoint_trigger_margin": 0.03,
            "phase11_waypoint_trigger_distance": 0.12,
            "phase11_obstacle_margin": 0.35,
            "phase11_waypoint_clearance": 0.45,
            "aim_degrees": 15.0,
            "standoff": 0.28,
            "lateral": 0.08,
            "lift_z": 0.20,
            "strike_frames": 18,
            "weight_shift": 0.025,
            "follow_x": 0.15,
            "settle_frames": 60,
            "recover_frames": 45,
            "forward_shift": 0.1,
            "foot_y": 0.0,
            "foot_yaw_deg": 0.0,
        }

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
        # Work on copies: the caller retains the true goal used by the scene
        # and scorer. Only this temporary reference direction is compensated.
        ball = np.asarray(obj_pos, dtype=np.float32).reshape(3).copy()
        goal = np.asarray(target_obj_pos, dtype=np.float32).reshape(3).copy()
        angle = np.deg2rad(self.cfg["aim_degrees"])
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        goal[:2] = ball[:2] + rotation @ (goal[:2] - ball[:2])
        parameters = {key: self.cfg[key] for key in (
            "standoff", "lateral", "lift_z", "strike_frames", "weight_shift",
            "follow_x", "settle_frames", "recover_frames", "forward_shift",
            "foot_y", "foot_yaw_deg",
        )}
        return generate_cartesian(
            self,
            pelvis_pos=np.asarray(pelvis_pos, dtype=np.float32).reshape(3),
            pelvis_quat=normalize_quat(pelvis_quat),
            obj_pos=ball,
            obj_quat=normalize_quat(obj_quat),
            box_half_dims=np.asarray(box_half_dims, dtype=np.float32).reshape(3),
            target_obj_pos=goal,
            task=task,
            parameters=parameters,
        )
