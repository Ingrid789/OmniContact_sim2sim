import numpy as np

from common.utils import quat_apply, yaw_quat, yaw_to_quat
from policy.omnicontact.CFgen_meta3_pushbox_innerside import CfGenPushBoxInnerSide
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_JOINT_POS_MJ,
    DEFAULT_PELVIS_Z,
    _append_contactloco_turn_walk_recover,
)


class CfGenPushBoxTwoSides(CfGenPushBoxInnerSide):
    """Rule-based two-side pushbox reference generator."""

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__(pad=pad, step_size_linear=step_size_linear, step_size_angular=step_size_angular)

        self.cfg.update(
            phase11_pre_push_standoff=0.4
        )

    def _get_target_face_info(self, obj_pos: np.ndarray, obj_quat: np.ndarray, pelvis_pos: np.ndarray, dims: np.ndarray):
        dx, dy, _ = [float(x) for x in np.asarray(dims, dtype=np.float32).reshape(3)]
        vec_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec_y = self._vec_y
        ax = quat_apply(obj_quat, vec_x)
        ay = quat_apply(obj_quat, vec_y)

        candidates = [
            (obj_pos + ax * dx, ax, dy),
            (obj_pos - ax * dx, -ax, dy),
            (obj_pos + ay * dy, ay, dx),
            (obj_pos - ay * dy, -ay, dx),
        ]
        pelvis_xy_ground = np.array([float(pelvis_pos[0]), float(pelvis_pos[1]), 0.0], dtype=np.float32)
        best = min(candidates, key=lambda x: float(np.linalg.norm(x[0] - pelvis_xy_ground)))
        face_pos, normal, half_width = best

        app_dir = -normal
        target_yaw = np.arctan2(app_dir[1], app_dir[0])
        target_quat = yaw_to_quat(target_yaw)
        return {"pos": face_pos, "n": normal, "w": half_width}, target_quat, target_yaw
    
    def _grasp_targets(self, obj_pos: np.ndarray, face: dict, target_yaw_quat: np.ndarray) -> dict[str, np.ndarray | None]:
        """Return task-specific grasp hand targets."""
        contact_center = obj_pos.copy()
        contact_half_width = float(face["w"]) * 1.5
        hand_axis = quat_apply(target_yaw_quat, self._vec_y).astype(np.float32)
        lw_contact = (contact_center + hand_axis * contact_half_width).astype(np.float32)
        rw_contact = (contact_center - hand_axis * contact_half_width).astype(np.float32)

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
        pelvis_pos0 = np.asarray(pelvis_pos, dtype=np.float32).copy()
        pelvis_quat0 = np.asarray(pelvis_quat, dtype=np.float32).copy()
        obj_pos0 = np.asarray(obj_pos, dtype=np.float32).copy()
        obj_quat0 = np.asarray(obj_quat, dtype=np.float32).copy()
        target_obj_pos = np.asarray(target_obj_pos, dtype=np.float32).copy()

        # Determine which face of the box to grasp based on the initial positions
        face, target_yaw_quat, target_yaw = self._get_target_face_info(obj_pos0, obj_quat0, pelvis_pos0, box_half_dims)
        b = self._new_builder()
        fwd13 = quat_apply(target_yaw_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        pelvis_p11_target = np.array(
            [
                obj_pos0[0] - fwd13[0] * self.cfg["phase11_pre_push_standoff"],
                obj_pos0[1] - fwd13[1] * self.cfg["phase11_pre_push_standoff"],
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )

        # -------------------------
        # Phase11: turn toward the standing point, walk there, then face the object.
        # -------------------------
        self._append_loco_approach_with_waypoints(
            b,
            pelvis_start=pelvis_pos0,
            pelvis_target=pelvis_p11_target,
            yaw_start=yaw_quat(pelvis_quat0),
            yaw_target=target_yaw_quat,
            step_linear=self.step_linear,
            step_angular=self.step_angular,
            object_pos=obj_pos0,
            object_quat=obj_quat0,
            obstacle_half_dims=box_half_dims,
            use_waypoints=False,
        )
        phase11_pelvis_pos = b.last("base_p").copy()
        phase11_pelvis_quat = b.last("base_q").copy()

        # -------------------------
        # Phase12: define the ending grasp/stance targets, solve IK, then interpolate to that optimized pose.
        # -------------------------
        obj_pos_push = 0.5 * (face['pos'] + obj_pos0)
        obj_pos_push[2] = 1.25 * box_half_dims[2]
        grasp_targets = self._grasp_targets(obj_pos_push, face, target_yaw_quat)
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
            object_pos=obj_pos_push,
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
        # Phase21: turn toward the push direction, contact-walk the box, then recover.
        # -------------------------
        o21_0 = b.last("obj_p")

        diff = target_obj_pos - o21_0
        if float(np.linalg.norm(diff[:2])) < 1e-6:
            yaw21_1 = b.last("base_q")
        else:
            yaw21_1 = yaw_to_quat(np.arctan2(float(diff[1]), float(diff[0]))).astype(np.float32)

        _append_contactloco_turn_walk_recover(
            b,
            phase_turn=21,
            phase_walk=22,
            phase_recover=23,
            yaw_target=yaw21_1,
            object_goal_pos=target_obj_pos,
            object_goal_standoff=0.05,
            step_angular=self.step_angular * 0.2,
            step_linear=self.step_linear * 0.5,
            contact=self._contact1_hands,
            walk_pad=self.pad,
            recover_pad=self.pad,
        )

        return b.finalize(), target_yaw
