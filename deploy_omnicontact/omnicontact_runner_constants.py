import numpy as np

LAB2MJ = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)

TORQUE_LIMIT = np.array(
    [
        88.0,
        88.0,
        88.0,
        139.0,
        139.0,
        50.0,
        88.0,
        88.0,
        50.0,
        139.0,
        139.0,
        25.0,
        25.0,
        50.0,
        50.0,
        25.0,
        25.0,
        50.0,
        50.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        5.0,
        5.0,
        5.0,
        5.0,
    ],
    dtype=np.float32,
)

CONTACT_COLOR_OFF = np.array([1.0, 1.0, 0.0, 0.7], dtype=np.float32)
CONTACT_COLOR_ON = np.array([1.0, 0.0, 0.0, 0.7], dtype=np.float32)
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
