from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


GO2_SCENE = (
    Path(__file__).resolve().parent
    / "third_party"
    / "mujoco_menagerie"
    / "unitree_go2"
    / "scene.xml"
)


JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


class Go2StandController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
            dtype=int,
        )
        self.qpos_ids = np.array([model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=int)
        self.qvel_ids = np.array([model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=int)

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise RuntimeError("Go2 model does not contain a 'home' keyframe")

        self.home_qpos = model.key_qpos[key_id].copy()
        self.q_des = self.home_qpos[self.qpos_ids].copy()
        self.kp = np.array([35.0, 45.0, 45.0] * 4)
        self.kd = np.array([1.2, 1.5, 1.5] * 4)

    def reset(self) -> None:
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.step()
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        q = self.data.qpos[self.qpos_ids]
        qd = self.data.qvel[self.qvel_ids]
        tau = self.kp * (self.q_des - q) - self.kd * qd
        self.data.ctrl[:] = np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, controller: Go2StandController, duration: float) -> None:
    steps = int(duration / model.opt.timestep)
    min_height = float("inf")

    for step in range(steps):
        controller.step()
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))

        if step % max(1, steps // 10) == 0:
            print(
                f"t={data.time:.2f} z={data.qpos[2]:.3f} "
                f"quat={np.round(data.qpos[3:7], 3)}"
            )

    print(f"done duration={data.time:.2f} final_z={data.qpos[2]:.3f} min_z={min_height:.3f}")


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, controller: Go2StandController) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 135

        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the clean MuJoCo Menagerie Unitree Go2 model.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless run duration in seconds.")
    args = parser.parse_args()

    if not GO2_SCENE.exists():
        raise FileNotFoundError(f"Go2 scene not found: {GO2_SCENE}")

    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data = mujoco.MjData(model)
    controller = Go2StandController(model, data)
    controller.reset()

    if args.headless:
        run_headless(model, data, controller, args.duration)
    else:
        run_viewer(model, data, controller)


if __name__ == "__main__":
    main()
