import os
from typing import Any

import numpy as np


REQUIRED_MOTION_FLOW_KEYS = {
    "joint_pos",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "contact_info",
    "ee_pos_w",
    "ee_quat_w",
    "body_pos_w",
    "body_quat_w",
}


def resolve_tracking_npz_path(policy: Any) -> str:
    npz_dir = str(getattr(policy, "npz_dir", "")).strip()
    if not npz_dir:
        raise ValueError(
            "reference_source=NPZmotion requires npz_dir. "
            "Please pass --npz-dir in runner."
        )

    npz_path = os.path.abspath(os.path.expanduser(npz_dir))
    if os.path.isfile(npz_path):
        return npz_path

    raise FileNotFoundError(f"Invalid npz_dir path: {npz_path}")


def _frame_slice(policy: Any, total_frames: int) -> slice:
    start = int(getattr(policy, "tracking_start_frame", 0))
    end = int(getattr(policy, "tracking_end_frame", -1))
    if start < 0:
        raise ValueError(f"tracking_start_frame must be >= 0, got {start}")
    if end < 0:
        end = total_frames
    if end > total_frames:
        raise ValueError(f"tracking_end_frame={end} exceeds total_frames={total_frames}")
    if start >= end:
        raise ValueError(
            f"Invalid tracking frame range: start={start}, end={end}, total_frames={total_frames}"
        )
    return slice(start, end)


def load_tracking_npz_reference(policy: Any) -> None:
    npz_path = resolve_tracking_npz_path(policy)
    npz_data = np.load(npz_path, allow_pickle=False)
    missing = sorted(REQUIRED_MOTION_FLOW_KEYS - set(npz_data.files))
    if missing:
        raise KeyError(f"tracking_npz requires motion_flow keys, missing: {missing}")

    total_frames = int(npz_data["joint_pos"].shape[0])
    frame_slice = _frame_slice(policy, total_frames)
    start = int(frame_slice.start)
    end = int(frame_slice.stop)

    policy.ref_joint_pos = np.asarray(npz_data["joint_pos"][frame_slice], dtype=np.float32)
    if "joint_vel" in npz_data:
        policy.ref_joint_vel = np.asarray(npz_data["joint_vel"][frame_slice], dtype=np.float32)
    else:
        policy.ref_joint_vel = np.zeros_like(policy.ref_joint_pos, dtype=np.float32)

    body_pos = np.asarray(npz_data["body_pos_w"][frame_slice], dtype=np.float32)
    body_quat = np.asarray(npz_data["body_quat_w"][frame_slice], dtype=np.float32)
    body_lin_vel = np.asarray(npz_data["body_lin_vel_w"][frame_slice], dtype=np.float32) if "body_lin_vel_w" in npz_data else None
    body_ang_vel = np.asarray(npz_data["body_ang_vel_w"][frame_slice], dtype=np.float32) if "body_ang_vel_w" in npz_data else None

    policy.ref_base_pos = body_pos[:, 0, :].astype(np.float32)
    policy.ref_base_quat = body_quat[:, 0, :].astype(np.float32)
    policy.ref_base_lin_vel = (
        body_lin_vel[:, 0, :].astype(np.float32)
        if body_lin_vel is not None
        else np.zeros((policy.ref_base_pos.shape[0], 3), dtype=np.float32)
    )
    policy.ref_base_ang_vel = (
        body_ang_vel[:, 0, :].astype(np.float32)
        if body_ang_vel is not None
        else np.zeros((policy.ref_base_pos.shape[0], 3), dtype=np.float32)
    )

    policy.ref_left_wrist_pos = body_pos[:, 37, :].astype(np.float32)
    policy.ref_left_wrist_quat = body_quat[:, 37, :].astype(np.float32)
    policy.ref_right_wrist_pos = body_pos[:, 38, :].astype(np.float32)
    policy.ref_right_wrist_quat = body_quat[:, 38, :].astype(np.float32)
    policy.ref_left_ankle_future_pos = body_pos[:, 25, :].astype(np.float32)
    policy.ref_left_ankle_future_quat = body_quat[:, 25, :].astype(np.float32)
    policy.ref_right_ankle_future_pos = body_pos[:, 26, :].astype(np.float32)
    policy.ref_right_ankle_future_quat = body_quat[:, 26, :].astype(np.float32)
    policy.ref_torso_future_pos = body_pos[:, 11, :].astype(np.float32)
    policy.ref_torso_future_quat = body_quat[:, 11, :].astype(np.float32)

    policy.ref_object_pos = np.asarray(npz_data["object_pos_w"][frame_slice], dtype=np.float32)
    policy.ref_object_quat = np.asarray(npz_data["object_quat_w"][frame_slice], dtype=np.float32)
    policy.ref_object_lin_vel = np.asarray(npz_data["object_lin_vel_w"][frame_slice], dtype=np.float32)
    policy.ref_object_ang_vel = np.asarray(npz_data["object_ang_vel_w"][frame_slice], dtype=np.float32)
    policy.ref_contact = np.asarray(npz_data["contact_info"][frame_slice], dtype=np.float32)

    policy.ref_table_1_pos = np.asarray(npz_data["table1_pos_w"][frame_slice], dtype=np.float32)
    policy.ref_table_2_pos = np.asarray(npz_data["table2_pos_w"][frame_slice], dtype=np.float32)

    print(
        f"[{policy.name_str}] loaded tracking_npz reference: {npz_path}, "
        f"frames=[{start}:{end}), nframes={policy.ref_base_pos.shape[0]}"
    )
