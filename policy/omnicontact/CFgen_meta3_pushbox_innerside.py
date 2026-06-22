import numpy as np

from common.utils import (
    quat_apply,
    yaw_quat,
    yaw_to_quat,
)
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_JOINT_POS_MJ,
    DEFAULT_PELVIS_Z,
    _append_contactloco_walk,
)
from policy.omnicontact.CFgen_base import CfGenBase


class CfGenPushBoxInnerSide(CfGenBase):
    """Rule-based inner-side pushbox reference generator."""
    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__()
        self.pad = int(pad)
        self.step_linear = float(step_size_linear)
        self.step_angular = float(step_size_angular)
        
        self.cfg = {
            "phase11_pre_push_standoff": 0.5,
            "phase2_push_hand_spacing": 0.2,
            "phase11_waypoint_trigger_margin": 0.05,
            "phase11_waypoint_trigger_distance": 0.18,
            "phase11_obstacle_margin": 0.4,
            "phase11_waypoint_clearance": 0.6,
            "courch_vel": 150.0,
            "ik_joint_indices": np.array(
                [
                    0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    14,
                    15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28,
                ],
                dtype=np.int32,
            ),
        }

    def _grasp_targets(self, obj_pos: np.ndarray, pelvis_pos: np.ndarray, target_yaw_quat: np.ndarray) -> dict[str, np.ndarray | None]:

        hand_half_spacing = 0.5 * float(self.cfg["phase2_push_hand_spacing"])
        local_lw_contact = np.array([0.2, +hand_half_spacing, 0])
        local_rw_contact = np.array([0.2, -hand_half_spacing, 0])

        lw_contact = (pelvis_pos + quat_apply(target_yaw_quat, local_lw_contact))
        rw_contact = (pelvis_pos + quat_apply(target_yaw_quat, local_rw_contact))
        lw_contact[2] = obj_pos[2]
        rw_contact[2] = obj_pos[2]

        return {"lw_contact": lw_contact, "rw_contact": rw_contact}

    def generate(
        self,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        obj_pos: np.ndarray,
        obj_quat: np.ndarray,
        box_half_dims: np.ndarray = np.array([0.2, 0.3, 0.15]),
        target_obj_pos: np.ndarray = np.array([1.0, 1.0, 0.5])
    ):
        b = self._new_builder()
        pelvis_pos0 = np.asarray(pelvis_pos, dtype=np.float32).copy()
        pelvis_quat0 = np.asarray((pelvis_quat), dtype=np.float32).copy()
        obj_pos0 = np.asarray(obj_pos, dtype=np.float32).copy()
        obj_pos0[2] = box_half_dims[2] # box on the ground in push task
        obj_quat0 = np.asarray((obj_quat), dtype=np.float32).copy()
        target_obj_pos = np.asarray(target_obj_pos, dtype=np.float32).copy()
        target_obj_pos[2] = box_half_dims[2] # box on the ground in push task

        move_dir = (target_obj_pos - obj_pos0).astype(np.float32)
        move_norm = float(np.linalg.norm(move_dir))
        move_dir = np.divide(move_dir, move_norm, out=np.zeros_like(move_dir), where=move_norm > 1e-6)
        target_yaw = float(np.arctan2(float(move_dir[1]), float(move_dir[0])))
        target_yaw_quat = yaw_to_quat(target_yaw).astype(np.float32)

        pre_push_standoff = float(self.cfg["phase11_pre_push_standoff"])
        pelvis_p12_target = np.array(
            [
                obj_pos0[0] - move_dir[0] * pre_push_standoff,
                obj_pos0[1] - move_dir[1] * pre_push_standoff,
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )

        # -------------------------
        # Phase11: turn toward the phase12 endpoint, walk there, then face the box.
        # -------------------------
        self._append_loco_approach_with_waypoints(
            b,
            pelvis_start=pelvis_pos0,
            pelvis_target=pelvis_p12_target,
            yaw_start=yaw_quat(pelvis_quat0),
            yaw_target=target_yaw_quat,
            step_linear=self.step_linear,
            step_angular=self.step_angular,
            object_pos=obj_pos0,
            object_quat=obj_quat0,
            obstacle_half_dims=box_half_dims,
        )
        phase11_pelvis_pos = b.last("base_p").copy()
        phase11_pelvis_quat = b.last("base_q").copy()

        # -------------------------
        # Phase12: define the ending grasp/stance targets, solve IK, then interpolate to that optimized pose.
        # -------------------------
        grasp_targets = self._grasp_targets(obj_pos0, b.last("base_p"), target_yaw_quat)
        lw_contact = grasp_targets["lw_contact"]
        rw_contact = grasp_targets["rw_contact"]

        q_phase12 = DEFAULT_JOINT_POS_MJ.copy()
        pelvis_contact, pelvis_quat_contact, q_contact = self._solve_fixed_feet_hand_ik(
            pelvis_pos=b.last("base_p"),
            pelvis_quat=b.last("base_q"),
            left_ankle_pos=b.last("la_p"),
            left_ankle_quat=b.last("la_q"),
            right_ankle_pos=b.last("ra_p"),
            right_ankle_quat=b.last("ra_q"),
            left_palm_pos=lw_contact,
            left_palm_quat=None,
            right_palm_pos=rw_contact,
            right_palm_quat=None,
            seed_joint_pos=q_phase12,
            object_pos=obj_pos0,
            **self.ik_params,
        )

        n12 = max(50, int(abs(pelvis_contact[2] - phase11_pelvis_pos[2]) * self.cfg["courch_vel"]))
        dof_seq12 = self._joint_lerp(b.last("dof_pos"), q_contact, n12)
        pelvis_seq12 = self._vec_lerp(phase11_pelvis_pos, pelvis_contact, n12)
        pelvis_quat_seq12 = self._quat_slerp_sequence(phase11_pelvis_quat, pelvis_quat_contact, n12)
        self._append_joint_fk_phase(
            b,
            12,
            obj_pos=obj_pos0,
            obj_quat=obj_quat0,
            qseq=dof_seq12,
            pelvis_seq=pelvis_seq12,
            pelvis_quat_seq=pelvis_quat_seq12,
            contact=self._contact0,
        )
        b.pad(12, contact=self._contact0, count=self.pad)
        
        # -------------------------
        # Phase21: contact push. Keep pelvis z at the current crouch height.
        # -------------------------
        delta21 = target_obj_pos - b.last("obj_p")

        _append_contactloco_walk(
            b,
            21,
            pelvis_start=b.last("base_p"),
            pelvis_target=b.last("base_p") + delta21,
            yaw=b.last("base_q"),
            step_linear=self.step_linear * 0.3,
            object_pos=b.last("obj_p"),
            object_quat=b.last("obj_q"),
            dof_pos=b.last("dof_pos"),
            contact=self._contact1_hands,
            keep_pelvis_z=True,
        )
        b.pad(21, contact=self._contact1_hands, count=self.pad)

        # -------------------------
        # Phase22: release contact and return to the default joint pose.
        # -------------------------
        yaw22 = yaw_quat(b.last("base_q"))

        pelvis_p22 = b.last("base_p").copy()
        pelvis_p22[2] = float(DEFAULT_PELVIS_Z)

        n22 = 60
        dof_seq22 = self._joint_lerp(b.last("dof_pos"), DEFAULT_JOINT_POS_MJ, n22)
        pelvis_pos_seq22 = self._vec_lerp(b.last("base_p"), pelvis_p22, n22)
        pelvis_quat_seq22 = self._quat_slerp_sequence(b.last("base_q"), yaw22, n22)

        self._append_joint_fk_phase(
            b,
            22,
            obj_pos=b.last("obj_p"),
            obj_quat=b.last("obj_q"),
            qseq=dof_seq22,
            pelvis_seq=pelvis_pos_seq22,
            pelvis_quat_seq=pelvis_quat_seq22,
            contact=self._contact0,
        )
        b.pad(22, contact=self._contact0, count=self.pad)

        return b.finalize(), target_yaw
