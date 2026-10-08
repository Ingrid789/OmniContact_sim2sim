"""Reachable kick trajectories, expressed in a ball-centred horizontal frame.

Only reference arrays are generated. The MuJoCo model below is the existing
kinematics model, never the live simulation. The left foot is constrained while
the pelvis transfers weight and the right foot follows a smooth strike path.
"""
from functools import lru_cache

import mujoco
import numpy as np

from common.utils import quat_apply_batch, yaw_quat, yaw_to_quat
from policy.omnicontact.CFgen_meta1_loco import (
    DEFAULT_JOINT_POS_MJ,
    DEFAULT_PELVIS_Z,
    KINEMATICS,
    _append_contactloco_recover,
    _append_fk_block,
    _append_loco_walk,
    _fk_reference_sequence_from_joints,
)


def smooth(a, b, n):
    """Cubic interpolation with zero endpoint velocity."""
    t = np.linspace(0, 1, n)
    t = t * t * (3 - 2 * t)
    return np.asarray(a)[None, :] * (1 - t[:, None]) + np.asarray(b)[None, :] * t[:, None]


@lru_cache(maxsize=256)
def canonical(
    standoff, lateral, foot_y, lift_z, strike_frames, weight_shift, follow_x,
    strike_height=0., forward_shift=0., support_step=0., foot_yaw_deg=0.,
):
    """Return pelvis/joint sequences with the ball at XY=0 and aim along +X.

    The optional support step and strike-height override allow reproducible
    comparisons with the development candidates. Production uses neither.
    Returned arrays are cached: callers must copy before modifying them.
    """
    m, d = KINEMATICS.model, KINEMATICS.data
    base = np.array([-standoff, lateral, DEFAULT_PELVIS_Z])
    q = DEFAULT_JOINT_POS_MJ.astype(float).copy()
    fk = KINEMATICS.forward(q, base, np.array([1., 0, 0, 0]))
    lf = fk['left_ankle_pitch_link']['pos']
    rf = fk['right_ankle_pitch_link']['pos']
    back = np.array([-.4, foot_y, lift_z])
    follow = np.array([follow_x, foot_y, .23])
    paths = np.concatenate([smooth(rf, back, 25), smooth(back, follow, strike_frames)[1:]], axis=0)
    if strike_height:
        paths[:25, 2] = smooth(np.array([rf[2]]), np.array([strike_height]), 25)[:, 0]
        rise = np.clip((paths[25:, 0] + .06) / (follow_x + .06), 0., 1.)
        rise = rise * rise * (3. - 2. * rise)
        paths[25:, 2] = strike_height + (.23 - strike_height) * rise
    bases = np.repeat(base[None, :], len(paths), axis=0)
    bases[:25, 1] += np.linspace(0, weight_shift, 25)
    bases[25:, 1] += weight_shift
    bases[:, 0] += np.linspace(0., forward_shift, len(paths))
    left_paths = np.repeat(lf[None, :], len(paths), axis=0)
    if support_step:
        planted = lf.copy()
        planted[0] += support_step
        left_paths[:] = planted
        u = np.linspace(0., 1., 35)
        stepping = smooth(lf, planted, 35)
        stepping[:, 2] += .05 * np.sin(np.pi * u) ** 2
        left_paths = np.concatenate([stepping, left_paths])
        paths = np.concatenate([np.tile(rf, (35, 1)), paths])
        preparation = np.tile(base, (35, 1))
        preparation[:, 1] -= .04 * np.sin(np.pi * u) ** 2
        bases = np.concatenate([preparation, bases])

    # The kinematics model has a free root followed by the twelve leg joints.
    # Solve both legs together; warm-starting each frame preserves continuity.
    jpos = np.empty((len(paths), len(q)))
    ids = [m.body(name).id for name in ('left_ankle_pitch_link', 'right_ankle_pitch_link')]
    limits = m.jnt_range[1:13]
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    orientation = np.zeros(3)
    for i, target in enumerate(paths):
        blend = np.clip((i - (35 if support_step else 0)) / 24., 0., 1.)
        blend = blend * blend * (3. - 2. * blend)
        right_orientation = yaw_to_quat(np.deg2rad(foot_yaw_deg) * blend).astype(float)
        d.qpos[:3] = bases[i]
        d.qpos[3:7] = [1, 0, 0, 0]
        for _ in range(35):
            d.qpos[7:] = q
            mujoco.mj_forward(m, d)
            errors, jacobians = [], []
            for body, pos, rotation in zip(
                ids, [left_paths[i], target], [np.array([1., 0, 0, 0]), right_orientation]
            ):
                mujoco.mj_jacBody(m, d, jacp, jacr, body)
                mujoco.mju_subQuat(orientation, rotation, d.xquat[body])
                errors.extend([pos - d.xpos[body], orientation * .08])
                jacobians.extend([jacp[:, 6:18].copy(), jacr[:, 6:18].copy() * .08])
            error = np.concatenate(errors)
            jac = np.concatenate(jacobians, axis=0)
            if np.linalg.norm(error) < 1e-5:
                break
            step = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(12) * 1e-5, error)
            q[:12] = np.clip(q[:12] + np.clip(step, -.15, .15), limits[:, 0], limits[:, 1])
        jpos[i] = q
    return bases, jpos


def generate_cartesian(
    generator, *, pelvis_pos, pelvis_quat, obj_pos, obj_quat, box_half_dims,
    target_obj_pos, task='kickball', parameters=None,
):
    """Attach a canonical single kick to an obstacle-aware walking approach."""
    cfg = dict(
        standoff=.28, lateral=.12, foot_y=0., lift_z=.15, strike_frames=18,
        weight_shift=.025, follow_x=.08, settle_frames=60, recover_frames=45,
        strike_height=0., forward_shift=0., support_step=0., foot_yaw_deg=0.,
    )
    cfg.update(parameters or {})
    direction, orientation, yaw = generator._move_direction(obj_pos, pelvis_pos, target_obj_pos)
    bases, joints = canonical(*(cfg[key] for key in (
        'standoff', 'lateral', 'foot_y', 'lift_z', 'strike_frames', 'weight_shift',
        'follow_x', 'strike_height', 'forward_shift', 'support_step', 'foot_yaw_deg',
    )))
    bases = quat_apply_batch(orientation, bases.astype(np.float32))
    bases[:, :2] += np.asarray(obj_pos)[:2]
    stance = bases[0].copy()
    intermediate = stance.copy()
    intermediate[:2] -= direction * .8
    b = generator._new_builder()
    generator._append_loco_approach_with_waypoints(
        b, phase_turn_to_walk=11, phase_walk=11, phase_turn_to_target=11,
        pelvis_start=pelvis_pos, pelvis_target=intermediate,
        yaw_start=yaw_quat(pelvis_quat), yaw_target=orientation,
        step_linear=generator.step_linear, step_angular=generator.step_angular,
        object_pos=obj_pos, object_quat=obj_quat, obstacle_half_dims=box_half_dims,
    )
    b.pad(11, contact=generator._contact0, count=generator.pad)
    _append_loco_walk(
        b, 12, pelvis_start=b.last('base_p'), pelvis_target=stance,
        yaw=orientation, step_linear=generator.step_linear * .75,
        object_pos=obj_pos, object_quat=obj_quat,
    )
    b.pad(12, contact=generator._contact0, count=cfg['settle_frames'])
    quats = np.tile(orientation, (len(bases), 1))
    lift_end = 25 + (35 if cfg['support_step'] else 0)
    for phase, segment in [(13, slice(0, lift_end)), (14, slice(lift_end, None))]:
        _append_fk_block(
            b, phase,
            fk_refs=_fk_reference_sequence_from_joints(bases[segment], quats[segment], joints[segment]),
            object_pos=obj_pos, object_quat=obj_quat,
            contact=generator._contact0 if phase == 13 else generator._contact_rfoot,
        )
    _append_contactloco_recover(
        b, 15, recover_frames=cfg['recover_frames'],
        recover_contact=generator._contact0, recover_pelvis_z=DEFAULT_PELVIS_Z,
    )
    b.pad(16, contact=generator._contact0, count=generator.pad)
    return b.finalize(), yaw
