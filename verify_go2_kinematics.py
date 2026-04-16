from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from quadruped_mj.go2_kinematics import GO2_LEGS


GO2_SCENE = (
    Path(__file__).resolve().parent
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_go2"
    / "scene.xml"
)


JOINT_SETS = {
    "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
    "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
    "RL": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
    "RR": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
}

FOOT_BODIES = {
    "FL": "FL_calf",
    "FR": "FR_calf",
    "RL": "RL_calf",
    "RR": "RR_calf",
}

FOOT_OFFSET = np.array([-0.002, 0.0, -0.213], dtype=float)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[key_id]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    print("Go2 kinematics verification against MuJoCo home pose:")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_pos = data.xpos[base_id].copy()
    base_rot = data.xmat[base_id].reshape(3, 3).copy()

    for leg_name, joint_names in JOINT_SETS.items():
        q = np.array(
            [
                data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
                for name in joint_names
            ],
            dtype=float,
        )
        kin = GO2_LEGS[leg_name]
        fk = kin.forward(q)

        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_BODIES[leg_name])
        mj_foot_world = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ FOOT_OFFSET
        mj_foot = base_rot.T @ (mj_foot_world - base_pos)

        q_ik = kin.inverse(mj_foot)
        fk_from_ik = kin.forward(q_ik)

        print(
            f"{leg_name}: "
            f"fk={np.round(fk, 5)} "
            f"mujoco={np.round(mj_foot, 5)} "
            f"fk_err={np.linalg.norm(fk - mj_foot):.6f} "
            f"ik_q={np.round(q_ik, 5)} "
            f"ik_err={np.linalg.norm(fk_from_ik - mj_foot):.6f}"
        )


if __name__ == "__main__":
    main()
