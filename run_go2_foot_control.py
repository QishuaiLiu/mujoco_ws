from __future__ import annotations

import argparse

import mujoco
import numpy as np

from quadruped_mj.go2_kinematics import GO2_LEGS
from run_go2 import GO2_SCENE, JOINT_NAMES, Go2StandController, run_viewer


LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_SLICES = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
FOOT_BODIES = {
    "FL": "FL_calf",
    "FR": "FR_calf",
    "RL": "RL_calf",
    "RR": "RR_calf",
}
FOOT_OFFSET = np.array([-0.002, 0.0, -0.213], dtype=float)


class Go2FootPositionController(Go2StandController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.foot_body_ids = {
            leg_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_BODIES[leg_name])
            for leg_name in LEG_NAMES
        }
        self.p_des_by_leg = {
            leg_name: GO2_LEGS[leg_name].forward(self.q_des[LEG_SLICES[leg_name]])
            for leg_name in LEG_NAMES
        }

    def step(self) -> None:
        self._update_joint_targets_from_feet()
        super().step()

    def _update_joint_targets_from_feet(self) -> None:
        for leg_name in LEG_NAMES:
            leg_slice = LEG_SLICES[leg_name]
            q_current = self.data.qpos[self.qpos_ids[leg_slice]]
            self.q_des[leg_slice] = GO2_LEGS[leg_name].inverse(
                self.p_des_by_leg[leg_name],
                initial_guess=q_current,
            )

    def set_desired_foot_position(self, leg_name: str, foot_position: np.ndarray) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        self.p_des_by_leg[leg_name] = np.asarray(foot_position, dtype=float).copy()

    def offset_desired_foot_position(self, leg_name: str, foot_offset: np.ndarray) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        self.p_des_by_leg[leg_name] = self.p_des_by_leg[leg_name] + np.asarray(foot_offset, dtype=float)

    def desired_foot_map(self) -> list[tuple[str, np.ndarray]]:
        return [(leg_name, self.p_des_by_leg[leg_name].copy()) for leg_name in LEG_NAMES]

    def actual_foot_position(self, leg_name: str) -> np.ndarray:
        leg_name = leg_name.upper()
        if leg_name not in self.foot_body_ids:
            raise KeyError(f"Unknown leg name: {leg_name}")

        base_pos = self.data.xpos[self.base_body_id]
        base_rot = self.data.xmat[self.base_body_id].reshape(3, 3)

        foot_body_id = self.foot_body_ids[leg_name]
        foot_body_pos = self.data.xpos[foot_body_id]
        foot_body_rot = self.data.xmat[foot_body_id].reshape(3, 3)

        foot_world = foot_body_pos + foot_body_rot @ FOOT_OFFSET
        return base_rot.T @ (foot_world - base_pos)

    def actual_foot_map(self) -> list[tuple[str, np.ndarray]]:
        return [(leg_name, self.actual_foot_position(leg_name)) for leg_name in LEG_NAMES]

    def foot_error_map(self) -> list[tuple[str, np.ndarray]]:
        return [
            (leg_name, self.p_des_by_leg[leg_name] - self.actual_foot_position(leg_name))
            for leg_name in LEG_NAMES
        ]

    def format_foot_tracking(self) -> str:
        lines = []
        for leg_name in LEG_NAMES:
            p_des = self.p_des_by_leg[leg_name]
            p_actual = self.actual_foot_position(leg_name)
            foot_error = p_des - p_actual
            lines.append(
                f"{leg_name} "
                f"p_des={np.round(p_des, 4)} "
                f"p_actual={np.round(p_actual, 4)} "
                f"err={np.round(foot_error, 4)}"
            )
        return "\n".join(lines)


def run_headless_foot_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Go2FootPositionController,
    duration: float,
    print_foot_every: int,
) -> None:
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

        if print_foot_every > 0 and controller.step_count % print_foot_every == 0:
            print(f"foot tracking t={data.time:.3f}")
            print(controller.format_foot_tracking())

    print(f"done duration={data.time:.2f} final_z={data.qpos[2]:.3f} min_z={min_height:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Go2 foot-position controller using IK + joint PD.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless run duration in seconds.")
    parser.add_argument(
        "--offset-foot",
        nargs=4,
        metavar=("LEG", "DX", "DY", "DZ"),
        help="Apply a base-frame foot offset in meters, for example: --offset-foot FL 0.01 0 0",
    )
    parser.add_argument(
        "--print-tau-every",
        type=int,
        default=0,
        help="Print torque command every N control steps. 0 disables repeated printing.",
    )
    parser.add_argument(
        "--print-foot-every",
        type=int,
        default=0,
        help="Print foot tracking every N control steps. 0 disables repeated printing.",
    )
    args = parser.parse_args()

    if not GO2_SCENE.exists():
        raise FileNotFoundError(f"Go2 scene not found: {GO2_SCENE}")

    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data = mujoco.MjData(model)
    controller = Go2FootPositionController(model, data)

    if args.offset_foot is not None:
        leg_name, dx, dy, dz = args.offset_foot
        controller.offset_desired_foot_position(
            leg_name,
            np.array([float(dx), float(dy), float(dz)], dtype=float),
        )

    controller.reset()

    print("Go2 desired foot targets (p_des in base frame):")
    for leg_name, foot_position in controller.desired_foot_map():
        print(f"  {leg_name}: {np.round(foot_position, 4)}")

    print("Go2 actual foot positions (p_actual in base frame):")
    for leg_name, foot_position in controller.actual_foot_map():
        print(f"  {leg_name}: {np.round(foot_position, 4)}")

    print("Go2 foot tracking error (p_des - p_actual):")
    for leg_name, foot_error in controller.foot_error_map():
        print(f"  {leg_name}: {np.round(foot_error, 4)}")

    print("Go2 desired joint targets from IK (q_des):")
    for joint_name, joint_value in controller.desired_joint_map():
        print(f"  {joint_name}: {joint_value:.4f}")

    print("Go2 initial torque command (tau):")
    for joint_name, tau_value in controller.torque_map():
        print(f"  {joint_name}: {tau_value:.4f}")

    if args.headless:
        run_headless_foot_control(
            model,
            data,
            controller,
            args.duration,
            args.print_foot_every,
        )
    else:
        run_viewer(model, data, controller, args.print_tau_every)


if __name__ == "__main__":
    main()
