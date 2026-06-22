import numpy as np

from common.utils import align_quat_hemisphere, quat_apply, quat_conjugate, quat_mul, quat_slerp
from policy.omnicontact.CFgen_meta2_carrybox import CfGenCarryBox
from policy.omnicontact.CFgen_meta3_pushbox_innerside import CfGenPushBoxInnerSide
from policy.omnicontact.CFgen_meta5_relocateball import CfGenRelocateBall


class CfGenPushCarryBox:
    """Meta-skill chain: push the push box first, then carry the carry box onto it."""

    PUSH_STAGE = "push_box"
    CARRY_STAGE = "carry_box"

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        self.push = CfGenPushBoxInnerSide(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )
        self.carry = CfGenCarryBox(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )

    def stage_plan(
        self,
        stage: str,
        *,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        carry_box_pos: np.ndarray,
        carry_box_quat: np.ndarray,
        push_box_dims: np.ndarray,
        carry_box_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> dict:
        push_box_dims = np.asarray(push_box_dims, dtype=np.float32).reshape(3)
        carry_box_dims = np.asarray(carry_box_dims, dtype=np.float32).reshape(3)

        if stage != self.CARRY_STAGE:
            goal = np.asarray(push_goal, dtype=np.float32).reshape(3).copy()
            goal[2] = float(push_box_dims[2])
            return {
                "stage": self.PUSH_STAGE,
                "generator": self.push,
                "object_name": "push_box",
                "box_dims": push_box_dims.copy(),
                "obj_pos": np.asarray(push_box_pos, dtype=np.float32).reshape(3).copy(),
                "obj_quat": np.asarray(push_box_quat, dtype=np.float32).reshape(4).copy(),
                "goal": goal,
            }

        push_box_pos = np.asarray(push_box_pos, dtype=np.float32).reshape(3)
        goal = push_box_pos.copy()
        goal[2] = float(push_box_pos[2] + push_box_dims[2] + carry_box_dims[2])
        return {
            "stage": self.CARRY_STAGE,
            "generator": self.carry,
            "object_name": "carry_box",
            "box_dims": carry_box_dims.copy(),
            "obj_pos": np.asarray(carry_box_pos, dtype=np.float32).reshape(3).copy(),
            "obj_quat": np.asarray(carry_box_quat, dtype=np.float32).reshape(4).copy(),
            "goal": goal,
        }

    def generate_stage(
        self,
        stage: str,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        carry_box_pos: np.ndarray,
        carry_box_quat: np.ndarray,
        push_box_dims: np.ndarray,
        carry_box_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> tuple[dict, dict, float]:
        plan = self.stage_plan(
            stage,
            push_box_pos=push_box_pos,
            push_box_quat=push_box_quat,
            carry_box_pos=carry_box_pos,
            carry_box_quat=carry_box_quat,
            push_box_dims=push_box_dims,
            carry_box_dims=carry_box_dims,
            push_goal=push_goal,
        )
        generator = plan["generator"]
        traj_data, target_yaw = generator.generate(
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            obj_pos=plan["obj_pos"],
            obj_quat=plan["obj_quat"],
            box_half_dims=plan["box_dims"],
            target_obj_pos=plan["goal"],
        )
        return plan, traj_data, float(target_yaw)


class CfGenCarryPushBox:
    """Meta-skill chain: carry the carry box onto a fixed plane, then push the push box under it."""

    CARRY_STAGE = "carry_box"
    PUSH_STAGE = "push_box"

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        self.carry = CfGenCarryBox(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )
        self.push = CfGenPushBoxInnerSide(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )

    @staticmethod
    def _plane_z(push_box_dims: np.ndarray) -> float:
        push_box_dims = np.asarray(push_box_dims, dtype=np.float32).reshape(3)
        return float(push_box_dims[2] * 2.0 + 0.02)

    def stage_plan(
        self,
        stage: str,
        *,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        carry_box_pos: np.ndarray,
        carry_box_quat: np.ndarray,
        push_box_dims: np.ndarray,
        carry_box_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> dict:
        push_box_dims = np.asarray(push_box_dims, dtype=np.float32).reshape(3)
        carry_box_dims = np.asarray(carry_box_dims, dtype=np.float32).reshape(3)
        plane_goal = np.asarray(push_goal, dtype=np.float32).reshape(3).copy()
        plane_z = self._plane_z(push_box_dims)

        if stage != self.PUSH_STAGE:
            goal = plane_goal.copy()
            goal[2] = float(plane_z + carry_box_dims[2])
            return {
                "stage": self.CARRY_STAGE,
                "generator": self.carry,
                "object_name": "carry_box",
                "box_dims": carry_box_dims.copy(),
                "obj_pos": np.asarray(carry_box_pos, dtype=np.float32).reshape(3).copy(),
                "obj_quat": np.asarray(carry_box_quat, dtype=np.float32).reshape(4).copy(),
                "goal": goal,
                "plane_z": plane_z,
            }

        goal = plane_goal.copy()
        goal[2] = float(push_box_dims[2])
        return {
            "stage": self.PUSH_STAGE,
            "generator": self.push,
            "object_name": "push_box",
            "box_dims": push_box_dims.copy(),
            "obj_pos": np.asarray(push_box_pos, dtype=np.float32).reshape(3).copy(),
            "obj_quat": np.asarray(push_box_quat, dtype=np.float32).reshape(4).copy(),
            "goal": goal,
            "plane_z": plane_z,
        }

    def generate_stage(
        self,
        stage: str,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        carry_box_pos: np.ndarray,
        carry_box_quat: np.ndarray,
        push_box_dims: np.ndarray,
        carry_box_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> tuple[dict, dict, float]:
        plan = self.stage_plan(
            stage,
            push_box_pos=push_box_pos,
            push_box_quat=push_box_quat,
            carry_box_pos=carry_box_pos,
            carry_box_quat=carry_box_quat,
            push_box_dims=push_box_dims,
            carry_box_dims=carry_box_dims,
            push_goal=push_goal,
        )
        traj_data, target_yaw = plan["generator"].generate(
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            obj_pos=plan["obj_pos"],
            obj_quat=plan["obj_quat"],
            box_half_dims=plan["box_dims"],
            target_obj_pos=plan["goal"],
        )
        return plan, traj_data, float(target_yaw)


class CfGenPushRelocateBall:
    """Meta-skill chain: push the cart first, then relocate one ball into it."""

    PUSH_STAGE = "push_box"
    RELOCATE_STAGE = "ball"

    def __init__(self, pad: int = 30, step_size_linear: float = 0.016, step_size_angular: float = 0.03):
        self.push = CfGenPushBoxInnerSide(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )
        self.relocate = CfGenRelocateBall(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )

    @staticmethod
    def _drop_z(push_box_dims: np.ndarray) -> float:
        push_box_dims = np.asarray(push_box_dims, dtype=np.float32).reshape(3)
        return float(push_box_dims[2] * 2.0)

    def stage_plan(
        self,
        stage: str,
        *,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
        push_box_dims: np.ndarray,
        ball_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> dict:
        push_box_dims = np.asarray(push_box_dims, dtype=np.float32).reshape(3)
        ball_dims = np.asarray(ball_dims, dtype=np.float32).reshape(3)

        if stage != self.RELOCATE_STAGE:
            goal = np.asarray(push_goal, dtype=np.float32).reshape(3).copy()
            goal[2] = float(push_box_dims[2])
            return {
                "stage": self.PUSH_STAGE,
                "generator": self.push,
                "object_name": "push_box",
                "box_dims": push_box_dims.copy(),
                "obj_pos": np.asarray(push_box_pos, dtype=np.float32).reshape(3).copy(),
                "obj_quat": np.asarray(push_box_quat, dtype=np.float32).reshape(4).copy(),
                "goal": goal,
            }

        goal = np.asarray(push_box_pos, dtype=np.float32).reshape(3).copy()
        goal[2] = self._drop_z(push_box_dims)
        return {
            "stage": self.RELOCATE_STAGE,
            "generator": self.relocate,
            "object_name": self.RELOCATE_STAGE,
            "box_dims": ball_dims.copy(),
            "obj_pos": np.asarray(ball_pos, dtype=np.float32).reshape(3).copy(),
            "obj_quat": np.asarray(ball_quat, dtype=np.float32).reshape(4).copy(),
            "goal": goal,
        }

    def generate_stage(
        self,
        stage: str,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        push_box_pos: np.ndarray,
        push_box_quat: np.ndarray,
        ball_pos: np.ndarray,
        ball_quat: np.ndarray,
        push_box_dims: np.ndarray,
        ball_dims: np.ndarray,
        push_goal: np.ndarray,
    ) -> tuple[dict, dict, float]:
        plan = self.stage_plan(
            stage,
            push_box_pos=push_box_pos,
            push_box_quat=push_box_quat,
            ball_pos=ball_pos,
            ball_quat=ball_quat,
            push_box_dims=push_box_dims,
            ball_dims=ball_dims,
            push_goal=push_goal,
        )
        traj_data, target_yaw = plan["generator"].generate(
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            obj_pos=plan["obj_pos"],
            obj_quat=plan["obj_quat"],
            box_half_dims=plan["box_dims"],
            target_obj_pos=plan["goal"],
        )
        return plan, traj_data, float(target_yaw)


class CfGenCarryCarryCarryBox:
    """Meta-skill chain: carry three boxes to the stack target, one stage at a time."""

    def __init__(self, pad: int = 30, step_size_linear: float = 0.014, step_size_angular: float = 0.03):
        self.carry = CfGenCarryBox(
            pad=pad,
            step_size_linear=step_size_linear,
            step_size_angular=step_size_angular,
        )

    @staticmethod
    def _local_pose(torso_pos: np.ndarray, torso_quat: np.ndarray, wrist_pos: np.ndarray, wrist_quat: np.ndarray):
        inv_torso = quat_conjugate(torso_quat)
        local_pos = quat_apply(inv_torso, wrist_pos - torso_pos).astype(np.float32)
        local_quat = quat_mul(inv_torso, wrist_quat).astype(np.float32)
        return local_pos, local_quat

    @staticmethod
    def _world_pose(torso_pos: np.ndarray, torso_quat: np.ndarray, local_pos: np.ndarray, local_quat: np.ndarray):
        wrist_pos = (torso_pos + quat_apply(torso_quat, local_pos)).astype(np.float32)
        wrist_quat = quat_mul(torso_quat, local_quat).astype(np.float32)
        wrist_quat = wrist_quat / (np.linalg.norm(wrist_quat) + 1e-8)
        return wrist_pos, wrist_quat.astype(np.float32)

    def _match_final_wrist_local_pose(self, traj: dict) -> dict:
        traj = {key: np.asarray(value).copy() for key, value in traj.items()}
        phase = np.asarray(traj["ref_phase"], dtype=np.int32).reshape(-1)
        final_phase = int(np.max(phase))
        final_idxs = np.flatnonzero(phase == final_phase)
        if final_idxs.size == 0:
            return traj

        t0_p = traj["ref_torso_future_pos"][0]
        t0_q = traj["ref_torso_future_quat"][0]
        lw_local_p, lw_local_q = self._local_pose(
            t0_p,
            t0_q,
            traj["ref_left_wrist_pos"][0],
            traj["ref_left_wrist_quat"][0],
        )
        rw_local_p, rw_local_q = self._local_pose(
            t0_p,
            t0_q,
            traj["ref_right_wrist_pos"][0],
            traj["ref_right_wrist_quat"][0],
        )

        n = int(final_idxs.size)
        u_seq = np.linspace(0.0, 1.0, n, dtype=np.float32)
        for local_p, local_q, pos_key, quat_key in (
            (lw_local_p, lw_local_q, "ref_left_wrist_pos", "ref_left_wrist_quat"),
            (rw_local_p, rw_local_q, "ref_right_wrist_pos", "ref_right_wrist_quat"),
        ):
            start_pos = traj[pos_key][final_idxs[0]].copy()
            start_quat = traj[quat_key][final_idxs[0]].copy()
            for j, idx in enumerate(final_idxs):
                u = float(u_seq[j])
                target_pos, target_quat = self._world_pose(
                    traj["ref_torso_future_pos"][idx],
                    traj["ref_torso_future_quat"][idx],
                    local_p,
                    local_q,
                )
                traj[pos_key][idx] = ((1.0 - u) * start_pos + u * target_pos).astype(np.float32)
                traj[quat_key][idx] = quat_slerp(start_quat, target_quat, u).astype(np.float32)

            traj[quat_key] = align_quat_hemisphere(traj[quat_key].astype(np.float32))

        return traj

    def stage_plan(
        self,
        stage_idx: int,
        *,
        box_pos: np.ndarray,
        box_quat: np.ndarray,
        box_dims: np.ndarray,
        box_goal_pos: np.ndarray,
        box_names: tuple[str, ...],
    ) -> dict:
        stage_idx = int(np.clip(stage_idx, 0, len(box_names) - 1))
        return {
            "stage_idx": stage_idx,
            "generator": self.carry,
            "object_name": box_names[stage_idx],
            "box_dims": np.asarray(box_dims[stage_idx], dtype=np.float32).reshape(3).copy(),
            "obj_pos": np.asarray(box_pos[stage_idx], dtype=np.float32).reshape(3).copy(),
            "obj_quat": np.asarray(box_quat[stage_idx], dtype=np.float32).reshape(4).copy(),
            "goal": np.asarray(box_goal_pos[stage_idx], dtype=np.float32).reshape(3).copy(),
        }

    def generate_stage(
        self,
        stage_idx: int,
        *,
        pelvis_pos: np.ndarray,
        pelvis_quat: np.ndarray,
        box_pos: np.ndarray,
        box_quat: np.ndarray,
        box_dims: np.ndarray,
        box_goal_pos: np.ndarray,
        box_names: tuple[str, ...],
    ) -> tuple[dict, dict, float]:
        plan = self.stage_plan(
            stage_idx,
            box_pos=box_pos,
            box_quat=box_quat,
            box_dims=box_dims,
            box_goal_pos=box_goal_pos,
            box_names=box_names,
        )
        traj_data, target_yaw = plan["generator"].generate(
            pelvis_pos=pelvis_pos,
            pelvis_quat=pelvis_quat,
            obj_pos=plan["obj_pos"],
            obj_quat=plan["obj_quat"],
            box_half_dims=plan["box_dims"],
            target_obj_pos=plan["goal"],
        )
        return plan, self._match_final_wrist_local_pose(traj_data), float(target_yaw)
