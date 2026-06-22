import numpy as np
from enum import Enum, unique

@unique
class FSMStateName(Enum):
    INVALID = -1
    PASSIVE = 1
    DEFAULTPOSE = 2
    SKILL_COOLDOWN = 3
    LOCOMODE = 4
    SKILL_SONIC = 11
    SKILL_OmniContact = 14
   

@unique
class FSMCommand(Enum):
    INVALID = -1
    POS_RESET = 1
    LOCO = 2
    PASSIVE = 4
    SKILL_3 = 7
    SKILL_SONIC = 12
    SKILL_OmniContact = 15
    
    
    

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def scale_values(values, target_ranges):
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled)



# ==================== Quaternion Utility Functions ====================

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return (q / (np.linalg.norm(q) + 1e-8)).astype(np.float32)


def align_quat_hemisphere(quat_seq: np.ndarray) -> np.ndarray:
    aligned = np.asarray(quat_seq, dtype=np.float32).copy()
    for i in range(1, len(aligned)):
        if float(np.dot(aligned[i - 1], aligned[i])) < 0.0:
            aligned[i] *= -1.0
    return aligned


def quat_conjugate(q):
    """Compute quaternion conjugate (inverse for unit quaternions)."""
    if isinstance(q, np.ndarray):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    else:
        w, x, y, z = q
        return np.array([w, -x, -y, -z], dtype=np.float32)

def quat_apply(q, v):
    v_shape = v.shape
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v, axis=-1) * np.expand_dims(q_w, axis=-1) * 2.0
    c = q_vec * \
        np.matmul(np.expand_dims(q_vec, axis=0), np.expand_dims(v, axis=-1)) * 2.0
    return (a + b + c).reshape(v_shape)

def quat_apply_batch(q, v):
    """Apply one quaternion (w, x, y, z) to a batch of vectors.

    Args:
        q: Quaternion with shape (4,).
        v: Vectors with shape (N, 3).

    Returns:
        Rotated vectors with shape (N, 3).
    """
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = 2.0 * q_w * np.cross(q_vec[None, :], v)
    c = 2.0 * np.sum(v * q_vec[None, :], axis=-1, keepdims=True) * q_vec[None, :]
    return (a + b + c).astype(np.float32)



def quat_mul_left_batch(q_left, q_right_batch):
    """Quaternion multiply q_left * q_right_batch.

    Args:
        q_left: Quaternion with shape (4,).
        q_right_batch: Quaternion batch with shape (N, 4).

    Returns:
        Result quaternion batch with shape (N, 4).
    """
    q_left = np.asarray(q_left, dtype=np.float32)
    q_right_batch = np.asarray(q_right_batch, dtype=np.float32)

    w1, x1, y1, z1 = q_left
    w2 = q_right_batch[:, 0]
    x2 = q_right_batch[:, 1]
    y2 = q_right_batch[:, 2]
    z2 = q_right_batch[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1).astype(np.float32)



def quat_to_6d_batch(q_batch):
    """Convert quaternion batch to 6D rotation representation.

    Args:
        q_batch: Quaternions with shape (N, 4), format (w, x, y, z).

    Returns:
        6D rotation representation with shape (N, 6), matching
        matrix_from_quat(q)[:, :2].reshape(-1) row-major layout.
    """
    q_batch = np.asarray(q_batch, dtype=np.float32)
    q_batch = q_batch / (np.linalg.norm(q_batch, axis=-1, keepdims=True) + 1e-8)

    w = q_batch[:, 0]
    x = q_batch[:, 1]
    y = q_batch[:, 2]
    z = q_batch[:, 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)

    return np.stack([r00, r01, r10, r11, r20, r21], axis=-1).astype(np.float32)


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation for 1 quaternion pair.

    Args:
        q0: Quaternion (w, x, y, z), shape (4,).
        q1: Quaternion (w, x, y, z), shape (4,).
        t:  Interpolation factor in [0, 1].
    """
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    t = float(t)

    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = float(np.dot(q0, q1))

    # Take shortest path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        res = q0 + t * (q1 - q0)
        return (res / (np.linalg.norm(res) + 1e-8)).astype(np.float32)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0 + s1 * q1).astype(np.float32)

def yaw_to_quat(yaw):
    """Convert yaw (radians) to quaternion (w, x, y, z) rotating about +Z."""
    yaw = np.asarray(yaw, dtype=np.float32)
    half = yaw * 0.5
    c = np.cos(half)
    s = np.sin(half)
    z0 = np.zeros_like(c)
    return np.stack([c, z0, z0, s], axis=-1).astype(np.float32)

def quat_rotate_inverse(q, v):
    """Rotate vector(s) by the inverse of quaternion q.

    Note: this helper assumes `q` is a single quaternion of shape (4,).
    """
    q = np.asarray(q, dtype=np.float32)
    q_conj = q.copy()
    q_conj[1:] *= -1.0
    return quat_apply(q_conj, np.asarray(v, dtype=np.float32))

def quat_mul(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([w, x, y, z])


def matrix_from_quat(q):
    """Convert quaternion to rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def subtract_frame_transforms(t01, q01, t02, q02):
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
    where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
        q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
            Defaults to None, in which case the position is assumed to be zero.
        q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            Defaults to None, in which case the orientation is assumed to be identity.

    Returns:
        A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
        Shape of the tensors are (N, 3) and (N, 4) respectively.
    """
    # compute orientation
    q10 = quat_conjugate(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def yaw_quat(q):
    """Extract yaw quaternion from full quaternion."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
