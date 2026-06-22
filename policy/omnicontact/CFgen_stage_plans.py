from typing import Any

import numpy as np


def carrybox_pushbox_plan(policy: Any, fk_info: dict):
    cfgen = policy.carry_push_cfgen if policy.task == "carry-push" else policy.push_carry_cfgen
    plan, traj_data, target_yaw = cfgen.generate_stage(
        policy.push_carry_stage,
        pelvis_pos=fk_info["pelvis"]["pos"],
        pelvis_quat=fk_info["pelvis"]["quat"],
        push_box_pos=policy.state_cmd.push_box_pos,
        push_box_quat=policy.state_cmd.push_box_quat,
        carry_box_pos=policy.state_cmd.carry_box_pos,
        carry_box_quat=policy.state_cmd.carry_box_quat,
        push_box_dims=policy.push_box_dims,
        carry_box_dims=policy.carry_box_dims,
        push_goal=policy.goal_pos,
    )
    policy.push_carry_stage = plan["stage"]
    policy.traj_generator = cfgen
    policy._set_active_object_profile(plan["object_name"], plan["box_dims"])
    return plan, traj_data, target_yaw


def carry_carry_carry_plan(policy: Any, fk_info: dict):
    stage_count = int(np.clip(getattr(policy, "stackbox_stage_count", 3), 1, len(policy.stack_box_names)))
    policy.traj_generator = policy.carry3_cfgen
    plan, traj_data, target_yaw = policy.carry3_cfgen.generate_stage(
        policy.stackbox_stage_idx,
        pelvis_pos=fk_info["pelvis"]["pos"],
        pelvis_quat=fk_info["pelvis"]["quat"],
        box_pos=policy.state_cmd.stack_box_pos[:stage_count],
        box_quat=policy.state_cmd.stack_box_quat[:stage_count],
        box_dims=policy.stack_box_dims[:stage_count],
        box_goal_pos=policy.stack_box_goal_pos[:stage_count],
        box_names=policy.stack_box_names[:stage_count],
    )
    policy.stackbox_stage_idx = plan["stage_idx"]
    policy._set_active_object_profile(plan["object_name"], plan["box_dims"])
    return plan, traj_data, target_yaw


def push_relocate_plan(policy: Any, fk_info: dict):
    plan, traj_data, target_yaw = policy.push_relocate_cfgen.generate_stage(
        policy.push_relocate_stage,
        pelvis_pos=fk_info["pelvis"]["pos"],
        pelvis_quat=fk_info["pelvis"]["quat"],
        push_box_pos=policy.state_cmd.push_box_pos,
        push_box_quat=policy.state_cmd.push_box_quat,
        ball_pos=policy.state_cmd.ball_pos,
        ball_quat=policy.state_cmd.ball_quat,
        push_box_dims=policy.push_box_dims,
        ball_dims=policy.ball_dims,
        push_goal=policy.goal_pos,
    )
    policy.push_relocate_stage = plan["stage"]
    policy.traj_generator = policy.push_relocate_cfgen
    policy._set_active_object_profile(plan["object_name"], plan["box_dims"])
    return plan, traj_data, target_yaw
