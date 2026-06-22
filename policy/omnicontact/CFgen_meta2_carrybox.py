import numpy as np

from common.utils import (
    quat_apply,
    yaw_quat,
    yaw_to_quat,
)
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_JOINT_POS_MJ,
    DEFAULT_PELVIS_Z,
    _append_contactloco_recover,
    _append_contactloco_turn_walk_recover,
    _append_loco_approach,
)
from policy.omnicontact.CFgen_base import CfGenBase


class CfGenCarryBox(CfGenBase):
    """Rule-based carrybox reference generator."""

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        super().__init__()
        self.pad = int(pad)
        self.step_linear = float(step_size_linear)
        self.step_angular = float(step_size_angular)

        # Nominal constants for the carrybox reference generator.
        self.cfg = {
            "phase11_pregrasp_standoff_dist": 0.4,
            "phase21_carry_object_z": 0.9,
            "phase22_object_goal_standoff": 0.4,
            "courch_vel": 150.0,
            "ik_joint_indices": np.array(
                [
                    0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    14,
                    15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 26, 28,
                ],
                dtype=np.int32,
            ),
        }


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
        contact_half_width = float(face["w"])
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
        target_obj_pos: np.ndarray = np.array([1., 1., 0.5]),
    ):
        pelvis_pos0 = np.asarray(pelvis_pos, dtype=np.float32).copy()
        pelvis_quat0 = np.asarray(pelvis_quat, dtype=np.float32).copy()
        obj_pos0 = np.asarray(obj_pos, dtype=np.float32).copy()
        obj_quat0 = np.asarray(obj_quat, dtype=np.float32).copy()
        target_obj_pos = np.asarray(target_obj_pos, dtype=np.float32).copy()

        # Determine which face of the box to grasp based on the initial positions
        face, target_yaw_quat, target_yaw = self._get_target_face_info(obj_pos0, obj_quat0, pelvis_pos0, box_half_dims)
        b = self._new_builder()
        fwd13 = quat_apply(target_yaw_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        face_pos = np.asarray(face["pos"], dtype=np.float32).reshape(3)
        pelvis_p11_target = np.array(
            [
                face_pos[0] - fwd13[0] * self.cfg["phase11_pregrasp_standoff_dist"],
                face_pos[1] - fwd13[1] * self.cfg["phase11_pregrasp_standoff_dist"],
                DEFAULT_PELVIS_Z,
            ],
            dtype=np.float32,
        )

        # -------------------------
        # Phase11: turn toward the standing point, walk there, then face the object.
        # -------------------------
        _append_loco_approach(
            b,
            pelvis_start=pelvis_pos0,
            pelvis_target=pelvis_p11_target,
            yaw_start=yaw_quat(pelvis_quat0),
            yaw_target=target_yaw_quat,
            step_linear=self.step_linear,
            step_angular=self.step_angular,
            object_pos=obj_pos0,
            object_quat=obj_quat0,
        )
        phase11_pelvis_pos = b.last("base_p").copy()
        phase11_pelvis_quat = b.last("base_q").copy()

        # -------------------------
        # Phase12: solve a contact upper-body IK, then interpolate to it.
        # -------------------------
        grasp_targets = self._grasp_targets(obj_pos0, face, target_yaw_quat)
        lw_contact = np.asarray(grasp_targets["lw_contact"], dtype=np.float32).reshape(3)
        rw_contact = np.asarray(grasp_targets["rw_contact"], dtype=np.float32).reshape(3)

        q_phase11 = DEFAULT_JOINT_POS_MJ.copy()
        pelvis_contact, pelvis_quat_contact, q_contact = self._solve_fixed_feet_hand_ik(
            pelvis_pos=b.last("base_p"),
            pelvis_quat=b.last("base_q"),
            left_ankle_pos=b.last("la_p"),
            left_ankle_quat=b.last("la_q"),
            right_ankle_pos=b.last("ra_p"),
            right_ankle_quat=b.last("ra_q"),
            left_palm_pos=lw_contact,
            right_palm_pos=rw_contact,
            seed_joint_pos=q_phase11,
            object_pos=obj_pos0,
            object_quat=obj_quat0,
            **self.ik_params,
        )

        n12 = max(50, int(abs(pelvis_contact[2] - phase11_pelvis_pos[2]) * self.cfg["courch_vel"]))
        dof_seq12 = self._joint_lerp(q_phase11, q_contact, n12)
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

        # -------------------------
        # Phase21: stand while carrying the object
        # -------------------------
        n21 = max(50, int(abs(pelvis_contact[2] - phase11_pelvis_pos[2]) * self.cfg["courch_vel"]))

        q21_target = q_contact.copy()
        q21_target[:15] = DEFAULT_JOINT_POS_MJ[:15]
        pelvis_p21_target = phase11_pelvis_pos.copy()
        pelvis_p21_target[2] = float(DEFAULT_PELVIS_Z)

        pelvis_quat_p21_seed = yaw_quat(pelvis_quat_contact).astype(np.float32)
        o21_target = pelvis_p21_target.copy()
        local_o21_xy = np.array([(2*face['w']), 0.0, 0.0], dtype=np.float32)
        o21_target[:2] = (pelvis_p21_target + quat_apply(pelvis_quat_p21_seed, local_o21_xy))[:2]
        o21_target[2] = float(self.cfg["phase21_carry_object_z"])
        hand_axis21 = (lw_contact - rw_contact).astype(np.float32)
        hand_axis21 /= max(float(np.linalg.norm(hand_axis21)), 1e-6)
        hand_half_width21 = float(face["w"])
        lw21_target = (o21_target + hand_axis21 * hand_half_width21).astype(np.float32)
        rw21_target = (o21_target - hand_axis21 * hand_half_width21).astype(np.float32)
        
        pelvis_p21_target, pelvis_quat_p21_target, q21_target = self._solve_fixed_feet_hand_ik(
            pelvis_pos=pelvis_p21_target,
            pelvis_quat=pelvis_quat_p21_seed,
            left_ankle_pos=b.last("la_p"),
            left_ankle_quat=b.last("la_q"),
            right_ankle_pos=b.last("ra_p"),
            right_ankle_quat=b.last("ra_q"),
            left_palm_pos=lw21_target,
            left_palm_quat=None,
            right_palm_pos=rw21_target,
            right_palm_quat=None,
            seed_joint_pos=q21_target,
            object_pos=o21_target,
            pelvis_z_min=DEFAULT_PELVIS_Z,
            **self.ik_params,
        )

        dof_seq21 = self._joint_lerp(q_contact, q21_target, n21)
        pelvis_seq21 = self._vec_lerp(pelvis_contact, pelvis_p21_target, n21)
        pelvis_quat_seq21 = self._quat_slerp_sequence(pelvis_quat_contact, pelvis_quat_p21_target, n21)

        o_p21 = self._vec_lerp(b.last("obj_p"), o21_target, n21)
        o_q21 = b.last("obj_q")

        self._append_joint_fk_phase(
            b,
            21,
            obj_pos=o_p21,
            obj_quat=o_q21,
            qseq=dof_seq21,
            pelvis_seq=pelvis_seq21,
            pelvis_quat_seq=pelvis_quat_seq21,
            contact=self._contact1_hands,
        )
        b.pad(21, contact=self._contact1_hands, count=self.pad)

        # -------------------------
        # Phase22: contact turn while keeping the carried object in the pelvis-local frame.
        # -------------------------
        o22_0 = b.last("obj_p")

        diff = target_obj_pos - o22_0
        if float(np.linalg.norm(diff[:2])) < 1e-6:
            yaw22_1 = b.last("base_q")
        else:
            yaw22_1 = yaw_to_quat(np.arctan2(float(diff[1]), float(diff[0]))).astype(np.float32)
        forward22 = quat_apply(yaw22_1, np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
        object_standoff23 = target_obj_pos.copy()
        object_standoff23[:2] -= forward22[:2] * float(self.cfg["phase22_object_goal_standoff"])

        _append_contactloco_turn_walk_recover(
            b,
            phase_turn=22,
            phase_walk=23,
            yaw_target=yaw22_1,
            object_target_pos=object_standoff23,
            object_goal_standoff=0.0,
            step_angular=self.step_angular,
            step_linear=self.step_linear,
            contact=self._contact1_hands,
            walk_pad=self.pad,
        )

        # -------------------------
        # Phase24: solve the final lowering pose
        # -------------------------
        pelvis_p24_0 = b.last("base_p")
        yaw24 = b.last("base_q")
        o24_0 = b.last("obj_p")
        oq24 = b.last("obj_q")
        q24_0 = b.last("dof_pos")

        o24_1 = target_obj_pos.copy()
        oq24_1 = yaw_quat(oq24)
        hand_axis24 = (b.last("lw_p") - b.last("rw_p")).astype(np.float32)
        hand_axis24 /= max(float(np.linalg.norm(hand_axis24)), 1e-6)
        hand_half_width24 = float(face["w"])
        lw24_1 = (o24_1 + hand_axis24 * hand_half_width24).astype(np.float32)
        rw24_1 = (o24_1 - hand_axis24 * hand_half_width24).astype(np.float32)
        pelvis_p24_1, pelvis_quat_p24_1, q24_1 = self._solve_fixed_feet_hand_ik(
            pelvis_pos=pelvis_p24_0,
            pelvis_quat=yaw24,
            left_ankle_pos=b.last("la_p"),
            left_ankle_quat=b.last("la_q"),
            right_ankle_pos=b.last("ra_p"),
            right_ankle_quat=b.last("ra_q"),
            left_palm_pos=lw24_1,
            left_palm_quat=None,
            right_palm_pos=rw24_1,
            right_palm_quat=None,
            seed_joint_pos=q24_0,
            object_pos=o24_1,
            object_quat=oq24_1,
            **self.ik_params,
        )

        n24 = max(50, int(abs(pelvis_p24_0[2] - pelvis_p24_1[2]) * self.cfg["courch_vel"]))
        dof_seq24 = self._joint_lerp(q24_0, q24_1, n24)
        pelvis_seq24 = self._vec_lerp(pelvis_p24_0, pelvis_p24_1, n24)
        pelvis_quat_seq24 = self._quat_slerp_sequence(yaw24, pelvis_quat_p24_1, n24)
        o_p24 = self._vec_lerp(o24_0, o24_1, n24)

        self._append_joint_fk_phase(
            b,
            24,
            obj_pos=o_p24,
            obj_quat=b.last("obj_q"),
            qseq=dof_seq24,
            pelvis_seq=pelvis_seq24,
            pelvis_quat_seq=pelvis_quat_seq24,
            contact=self._contact1_hands,
        )

        # -------------------------
        # Phase25: return to default dof while preserving the current ankle positions.
        # -------------------------
        _append_contactloco_recover(
            b,
            25,
            recover_frames=60,
            recover_contact=self._contact0,
            recover_pad=self.pad,
        )

        return b.finalize(), target_yaw
