import mujoco
import numpy as np

class MujocoKinematics:
    def __init__(self, xml_path: str, joint_names: list[str] = None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.has_free_joint = self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.qpos_offset = 7 if self.has_free_joint else 0
        self.num_bodies = self.model.nbody
        self.body_offset = 1 if self.has_free_joint else 0

        body_names = []
        for i in range(self.body_offset, self.num_bodies):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.append(name)
        self.body_names = body_names

        self.update_joint_names_subset(joint_names)

    def update_joint_names_subset(self, joint_names_subset: list[str] | None = None):
        if joint_names_subset is not None:
            self.joint_names = joint_names_subset
        else:
            self.joint_names = []
            for i in range(self.model.njnt):
                # skip root joint
                if self.has_free_joint and i == 0:
                    continue
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                self.joint_names.append(joint_name)

        self.num_joints = len(self.joint_names)

        self.joint_qpos_indices = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in the model.")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.joint_qpos_indices.append(qpos_addr)

    def forward(
        self,
        joint_pos: np.ndarray,
        base_pos: np.ndarray | None = None,
        base_quat: np.ndarray | None = None,
        joint_vel: np.ndarray | None = None,
        base_lin_vel: np.ndarray | None = None,
        base_ang_vel: np.ndarray | None = None,
    ) -> dict:
        # ---------------- qpos ----------------
        qpos_full = np.zeros(self.model.nq, dtype=np.float64)

        if self.has_free_joint:
            if base_pos is not None:
                qpos_full[0:3] = base_pos

            if base_quat is not None:
                qpos_full[3:7] = base_quat # Assume input is [w, x, y, z]

        qpos_full[self.joint_qpos_indices] = joint_pos
        self.data.qpos[:] = qpos_full

        # ---------------- qvel ----------------
        if joint_vel is not None or base_lin_vel is not None or base_ang_vel is not None:
            qvel_full = np.zeros(self.model.nv, dtype=np.float64)
            if self.has_free_joint:
                if base_lin_vel is not None:
                    qvel_full[0:3] = base_lin_vel
                if base_ang_vel is not None:
                    qvel_full[3:6] = base_ang_vel
                offset = 6
            else:
                offset = 0
            if joint_vel is not None:
                qvel_full[offset : offset + self.num_joints] = joint_vel
            self.data.qvel[:] = qvel_full

        # ---------------- forward ----------------
        mujoco.mj_forward(self.model, self.data)

        # ---------------- body info ----------------
        nbody = self.model.nbody
        offset = 1 if self.has_free_joint else 0
        body_info = {}
        for i in range(offset, nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            pos = self.data.xpos[i].copy()
            quat = self.data.xquat[i].copy() # [w, x, y, z]
            lin_vel = self.data.cvel[i].copy()[3:]
            ang_vel = self.data.cvel[i].copy()[0:3]
            body_info[name] = dict(
                pos=pos,
                quat=quat,
                lin_vel=lin_vel,
                ang_vel=ang_vel,
            )

        return body_info

