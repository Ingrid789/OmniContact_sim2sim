import numpy as np

from common.utils import align_quat_hemisphere, normalize_quat, quat_conjugate, quat_mul, yaw_quat
from policy.omnicontact.CFgen_meta1_loco import DEFAULT_JOINT_POS_MJ, KINEMATICS, _quat_to_rpy_deg


class _TrajBuilder:
    """Append phase blocks and finalize OmniContact reference arrays."""

    def __init__(self):
        self.lw_p: list[np.ndarray] = []
        self.lw_q: list[np.ndarray] = []
        self.rw_p: list[np.ndarray] = []
        self.rw_q: list[np.ndarray] = []
        self.obj_p: list[np.ndarray] = []
        self.obj_q: list[np.ndarray] = []
        self.contact: list[np.ndarray] = []
        self.phase: list[np.ndarray] = []
        self.base_p: list[np.ndarray] = []
        self.base_q: list[np.ndarray] = []

        self.torso_p: list[np.ndarray] = []
        self.torso_yaw_q: list[np.ndarray] = []
        self.torso_pitch_deg: list[np.ndarray] = []
        self.la_p: list[np.ndarray] = []
        self.la_q: list[np.ndarray] = []
        self.ra_p: list[np.ndarray] = []
        self.ra_q: list[np.ndarray] = []
        self.dof_pos: list[np.ndarray] = []

    @staticmethod
    def _bcast(x: np.ndarray, n: int, tail: tuple[int, ...]) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, *tail)
        a = a.reshape(-1, *tail)
        if len(a) == 1 and n > 1:
            a = np.repeat(a, n, axis=0)
        return a.astype(np.float32)

    @staticmethod
    def _bcast_1d(x: np.ndarray, n: int) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(a) == 1 and n > 1:
            a = np.full(n, float(a[0]), dtype=np.float32)
        return a.astype(np.float32)

    @staticmethod
    def _bcast_contact(x: np.ndarray, n: int) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, 4)
        a = a.reshape(-1, 4)
        if len(a) == 1 and n > 1:
            a = np.repeat(a, n, axis=0)
        return a.astype(np.float32)

    @staticmethod
    def _infer_dof_from_torso_pitch(torso_pitch_deg: np.ndarray, n: int) -> np.ndarray:
        dof = np.tile(DEFAULT_JOINT_POS_MJ.reshape(1, -1), (n, 1)).astype(np.float32)
        pitch_rad = np.deg2rad(np.asarray(torso_pitch_deg, dtype=np.float32).reshape(n))
        if dof.shape[1] > 14:
            dof[:, 14] = pitch_rad
        return dof

    @staticmethod
    def _infer_base_from_torso(torso_p: np.ndarray, base_q: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        torso_p = np.asarray(torso_p, dtype=np.float32).reshape(-1, 3)
        base_q = np.asarray(base_q, dtype=np.float32).reshape(-1, 4)
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(-1, len(DEFAULT_JOINT_POS_MJ))
        base_p = np.zeros_like(torso_p, dtype=np.float32)
        for i, q in enumerate(dof_pos):
            fk = KINEMATICS.forward(q, np.zeros(3, dtype=np.float32), base_q[i])
            base_p[i] = (torso_p[i] - fk["torso_link"]["pos"]).astype(np.float32)
        return base_p

    @staticmethod
    def _fk_refs(base_p: np.ndarray, base_q: np.ndarray, dof_pos: np.ndarray) -> dict[str, np.ndarray]:
        base_p = np.asarray(base_p, dtype=np.float32).reshape(-1, 3)
        base_q = align_quat_hemisphere(np.asarray(base_q, dtype=np.float32).reshape(-1, 4))
        dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(-1, len(DEFAULT_JOINT_POS_MJ))
        n = int(len(dof_pos))
        refs = {
            "lw_p": np.zeros((n, 3), dtype=np.float32),
            "lw_q": np.zeros((n, 4), dtype=np.float32),
            "rw_p": np.zeros((n, 3), dtype=np.float32),
            "rw_q": np.zeros((n, 4), dtype=np.float32),
            "torso_p": np.zeros((n, 3), dtype=np.float32),
            "torso_yaw_q": np.zeros((n, 4), dtype=np.float32),
            "torso_pitch_deg": np.zeros(n, dtype=np.float32),
            "la_p": np.zeros((n, 3), dtype=np.float32),
            "la_q": np.zeros((n, 4), dtype=np.float32),
            "ra_p": np.zeros((n, 3), dtype=np.float32),
            "ra_q": np.zeros((n, 4), dtype=np.float32),
        }
        for i, q in enumerate(dof_pos):
            fk = KINEMATICS.forward(q, base_p[i], base_q[i])
            torso_quat = normalize_quat(fk["torso_link"]["quat"])
            torso_yaw = normalize_quat(yaw_quat(torso_quat))
            torso_rel = normalize_quat(quat_mul(quat_conjugate(torso_yaw), torso_quat))
            _, pitch_deg, _ = _quat_to_rpy_deg(torso_rel)
            refs["lw_p"][i] = fk["left_palm_link"]["pos"]
            refs["lw_q"][i] = fk["left_palm_link"]["quat"]
            refs["rw_p"][i] = fk["right_palm_link"]["pos"]
            refs["rw_q"][i] = fk["right_palm_link"]["quat"]
            refs["torso_p"][i] = fk["torso_link"]["pos"]
            refs["torso_yaw_q"][i] = torso_yaw
            refs["torso_pitch_deg"][i] = float(pitch_deg)
            refs["la_p"][i] = fk["left_ankle_pitch_link"]["pos"]
            refs["la_q"][i] = fk["left_ankle_pitch_link"]["quat"]
            refs["ra_p"][i] = fk["right_ankle_pitch_link"]["pos"]
            refs["ra_q"][i] = fk["right_ankle_pitch_link"]["quat"]
        refs["lw_q"] = align_quat_hemisphere(refs["lw_q"])
        refs["rw_q"] = align_quat_hemisphere(refs["rw_q"])
        refs["torso_yaw_q"] = align_quat_hemisphere(refs["torso_yaw_q"])
        refs["la_q"] = align_quat_hemisphere(refs["la_q"])
        refs["ra_q"] = align_quat_hemisphere(refs["ra_q"])
        return refs

    def append(
        self,
        phase: int,
        *,
        lw_p: np.ndarray,
        lw_q: np.ndarray,
        rw_p: np.ndarray,
        rw_q: np.ndarray,
        obj_p: np.ndarray,
        obj_q: np.ndarray,
        torso_p: np.ndarray,
        torso_yaw_q: np.ndarray,
        torso_pitch_deg: np.ndarray,
        la_p: np.ndarray,
        ra_p: np.ndarray,
        la_q: np.ndarray | None = None,
        ra_q: np.ndarray | None = None,
        dof_pos: np.ndarray | None = None,
        base_p: np.ndarray | None = None,
        base_q: np.ndarray | None = None,
        contact: np.ndarray,
    ) -> None:
        torso_p = np.asarray(torso_p, dtype=np.float32).reshape(-1, 3)
        n = int(len(torso_p))
        torso_yaw_q = self._bcast(torso_yaw_q, n, (4,))
        torso_pitch_deg = self._bcast_1d(torso_pitch_deg, n)
        dof_pos = (
            self._infer_dof_from_torso_pitch(torso_pitch_deg, n)
            if dof_pos is None
            else self._bcast(dof_pos, n, (len(DEFAULT_JOINT_POS_MJ),))
        )
        base_q = torso_yaw_q if base_q is None else self._bcast(base_q, n, (4,))
        base_p = self._infer_base_from_torso(torso_p, base_q, dof_pos) if base_p is None else self._bcast(base_p, n, (3,))
        fk_refs = self._fk_refs(base_p, base_q, dof_pos)

        self.lw_p.append(fk_refs["lw_p"])
        self.lw_q.append(fk_refs["lw_q"])
        self.rw_p.append(fk_refs["rw_p"])
        self.rw_q.append(fk_refs["rw_q"])
        self.obj_p.append(self._bcast(obj_p, n, (3,)))
        self.obj_q.append(self._bcast(obj_q, n, (4,)))
        self.base_p.append(base_p.astype(np.float32))
        self.base_q.append(align_quat_hemisphere(base_q.astype(np.float32)))
        self.torso_p.append(fk_refs["torso_p"])
        self.torso_yaw_q.append(fk_refs["torso_yaw_q"])
        self.torso_pitch_deg.append(fk_refs["torso_pitch_deg"])
        self.la_p.append(fk_refs["la_p"])
        self.la_q.append(fk_refs["la_q"])
        self.ra_p.append(fk_refs["ra_p"])
        self.ra_q.append(fk_refs["ra_q"])
        self.dof_pos.append(dof_pos.astype(np.float32))
        self.contact.append(self._bcast_contact(contact, n))
        self.phase.append(np.full(n, int(phase), dtype=np.int32))


    def pad(self, phase: int, *, contact: np.ndarray, count: int) -> None:
        if count <= 0:
            return

        def _pad_last(arr_list: list[np.ndarray]) -> None:
            last = np.asarray(arr_list[-1][-1], dtype=np.float32).reshape(1, -1)
            arr_list.append(np.repeat(last, count, axis=0).astype(np.float32))

        def _pad_last_scalar(arr_list: list[np.ndarray]) -> None:
            last = float(arr_list[-1][-1])
            arr_list.append(np.full(count, last, dtype=np.float32))

        _pad_last(self.lw_p)
        _pad_last(self.lw_q)
        _pad_last(self.rw_p)
        _pad_last(self.rw_q)
        _pad_last(self.obj_p)
        _pad_last(self.obj_q)
        if self.base_p:
            _pad_last(self.base_p)
        if self.base_q:
            _pad_last(self.base_q)
        _pad_last(self.torso_p)
        _pad_last(self.torso_yaw_q)
        _pad_last_scalar(self.torso_pitch_deg)
        _pad_last(self.la_p)
        _pad_last(self.la_q)
        _pad_last(self.ra_p)
        _pad_last(self.ra_q)
        if self.dof_pos:
            _pad_last(self.dof_pos)
        self.contact.append(np.tile(np.asarray(contact, dtype=np.float32).reshape(1, 4), (count, 1)))
        self.phase.append(np.full(count, int(phase), dtype=np.int32))

    def last(self, name: str) -> np.ndarray:
        m = {
            "lw_p": self.lw_p,
            "lw_q": self.lw_q,
            "rw_p": self.rw_p,
            "rw_q": self.rw_q,
            "obj_p": self.obj_p,
            "obj_q": self.obj_q,
            "base_p": self.base_p,
            "base_q": self.base_q,
            "torso_p": self.torso_p,
            "torso_yaw_q": self.torso_yaw_q,
            "torso_pitch_deg": self.torso_pitch_deg,
            "la_p": self.la_p,
            "la_q": self.la_q,
            "ra_p": self.ra_p,
            "ra_q": self.ra_q,
            "dof_pos": self.dof_pos,
        }
        if name not in m or not m[name]:
            raise KeyError(name)
        return np.asarray(m[name][-1][-1]).copy()

    def finalize(self) -> dict:
        ref_left_wrist_pos = np.concatenate(self.lw_p, axis=0).astype(np.float32)
        ref_right_wrist_pos = np.concatenate(self.rw_p, axis=0).astype(np.float32)
        ref_left_wrist_quat = align_quat_hemisphere(np.concatenate(self.lw_q, axis=0).astype(np.float32))
        ref_right_wrist_quat = align_quat_hemisphere(np.concatenate(self.rw_q, axis=0).astype(np.float32))
        ref_object_pos = np.concatenate(self.obj_p, axis=0).astype(np.float32)
        ref_object_quat = align_quat_hemisphere(np.concatenate(self.obj_q, axis=0).astype(np.float32))
        ref_contact = np.concatenate(self.contact, axis=0).astype(np.float32)
        ref_phase = np.concatenate(self.phase, axis=0).astype(np.int32).reshape(-1)
        ref_base_pos = np.concatenate(self.base_p, axis=0).astype(np.float32) if self.base_p else None
        ref_base_quat = (
            align_quat_hemisphere(np.concatenate(self.base_q, axis=0).astype(np.float32))
            if self.base_q
            else None
        )

        ref_torso_pos = np.concatenate(self.torso_p, axis=0).astype(np.float32)
        ref_yaw_quat = align_quat_hemisphere(np.concatenate(self.torso_yaw_q, axis=0).astype(np.float32))
        pitch_deg = np.concatenate(self.torso_pitch_deg, axis=0).astype(np.float32).reshape(-1)
        ref_left_ankle_pos = np.concatenate(self.la_p, axis=0).astype(np.float32)
        ref_left_ankle_quat = align_quat_hemisphere(np.concatenate(self.la_q, axis=0).astype(np.float32))
        ref_right_ankle_pos = np.concatenate(self.ra_p, axis=0).astype(np.float32)
        ref_right_ankle_quat = align_quat_hemisphere(np.concatenate(self.ra_q, axis=0).astype(np.float32))

        pitch_rad = np.deg2rad(pitch_deg)
        pitch_quat = np.stack(
            [
                np.cos(0.5 * pitch_rad),
                np.zeros_like(pitch_rad),
                np.sin(0.5 * pitch_rad),
                np.zeros_like(pitch_rad),
            ],
            axis=-1,
        ).astype(np.float32)
        ref_torso_quat = np.array([quat_mul(y, p) for y, p in zip(ref_yaw_quat, pitch_quat)], dtype=np.float32)
        ref_torso_quat = align_quat_hemisphere(ref_torso_quat)

        traj = {
            "ref_left_wrist_pos": ref_left_wrist_pos,
            "ref_left_wrist_quat": ref_left_wrist_quat,
            "ref_right_wrist_pos": ref_right_wrist_pos,
            "ref_right_wrist_quat": ref_right_wrist_quat,
            "ref_object_pos": ref_object_pos,
            "ref_object_quat": ref_object_quat,
            "ref_contact": ref_contact,
            "ref_phase": ref_phase,
            "ref_torso_future_pos": ref_torso_pos,
            "ref_torso_future_quat": ref_torso_quat,
            "ref_left_ankle_future_pos": ref_left_ankle_pos,
            "ref_left_ankle_future_quat": ref_left_ankle_quat,
            "ref_right_ankle_future_pos": ref_right_ankle_pos,
            "ref_right_ankle_future_quat": ref_right_ankle_quat,
        }
        if self.dof_pos:
            traj["dof_pos"] = np.concatenate(self.dof_pos, axis=0).astype(np.float32)
        if ref_base_pos is not None:
            traj["ref_base_pos"] = ref_base_pos
        if ref_base_quat is not None:
            traj["ref_base_quat"] = ref_base_quat
        return traj
