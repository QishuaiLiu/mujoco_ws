from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from quadruped_mj.go2_kinematics import GO2_LEGS
from run_go2 import GO2_SCENE, JOINT_NAMES, Go2StandController


LEG_NAMES = ["FL", "FR", "RL", "RR"]
DIAGONAL_PAIRS = {
    "FL_RR": ("FL", "RR"),
    "FR_RL": ("FR", "RL"),
}
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
        self.p_nominal_by_leg = {
            leg_name: foot_position.copy()
            for leg_name, foot_position in self.p_des_by_leg.items()
        }
        self.lift_leg_name: str | None = None
        self.lift_amplitude = 0.0
        self.lift_frequency = 0.0
        self.swing_leg_name: str | None = None
        self.swing_length = 0.0
        self.swing_amplitude = 0.0
        self.swing_frequency = 0.0
        self.swing_diagonal_name: str | None = None

    def step(self) -> None:
        self._update_scripted_foot_targets()
        self._update_joint_targets_from_feet()
        super().step()

    def configure_lift_motion(self, leg_name: str, amplitude: float, frequency: float) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        if amplitude < 0.0:
            raise ValueError("Lift amplitude must be non-negative")
        if frequency <= 0.0:
            raise ValueError("Lift frequency must be positive")
        self.lift_leg_name = leg_name
        self.lift_amplitude = float(amplitude)
        self.lift_frequency = float(frequency)

    def configure_swing_motion(
        self,
        leg_name: str,
        length: float,
        amplitude: float,
        frequency: float,
    ) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        if length < 0.0:
            raise ValueError("Swing length must be non-negative")
        if amplitude < 0.0:
            raise ValueError("Swing amplitude must be non-negative")
        if frequency <= 0.0:
            raise ValueError("Swing frequency must be positive")
        self.swing_leg_name = leg_name
        self.swing_length = float(length)
        self.swing_amplitude = float(amplitude)
        self.swing_frequency = float(frequency)

    def configure_diagonal_swing_motion(
        self,
        diagonal_name: str,
        length: float,
        amplitude: float,
        frequency: float,
    ) -> None:
        diagonal_name = diagonal_name.upper()
        if diagonal_name not in DIAGONAL_PAIRS:
            raise KeyError(f"Unknown diagonal pair: {diagonal_name}")
        if length < 0.0:
            raise ValueError("Swing length must be non-negative")
        if amplitude < 0.0:
            raise ValueError("Swing amplitude must be non-negative")
        if frequency <= 0.0:
            raise ValueError("Swing frequency must be positive")
        self.swing_diagonal_name = diagonal_name
        self.swing_length = float(length)
        self.swing_amplitude = float(amplitude)
        self.swing_frequency = float(frequency)

    def _update_scripted_foot_targets(self) -> None:
        if self.lift_leg_name is None:
            pass
        else:
            phase = 2.0 * np.pi * self.lift_frequency * self.data.time
            z_lift = 0.5 * self.lift_amplitude * (1.0 - np.cos(phase))
            target = self.p_nominal_by_leg[self.lift_leg_name].copy()
            target[2] += z_lift
            self.p_des_by_leg[self.lift_leg_name] = target

        if self.swing_leg_name is None and self.swing_diagonal_name is None:
            return
        phase = 2.0 * np.pi * self.swing_frequency * self.data.time
        x_swing = 0.5 * self.swing_length * np.sin(phase)
        z_lift = 0.5 * self.swing_amplitude * (1.0 - np.cos(phase))
        swing_legs = []
        if self.swing_leg_name is not None:
            swing_legs.append(self.swing_leg_name)
        if self.swing_diagonal_name is not None:
            swing_legs.extend(DIAGONAL_PAIRS[self.swing_diagonal_name])
        for leg_name in swing_legs:
            target = self.p_nominal_by_leg[leg_name].copy()
            target[0] += x_swing
            target[2] += z_lift
            self.p_des_by_leg[leg_name] = target

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
        self.p_nominal_by_leg[leg_name] = np.asarray(foot_position, dtype=float).copy()
        self.p_des_by_leg[leg_name] = self.p_nominal_by_leg[leg_name].copy()

    def offset_desired_foot_position(self, leg_name: str, foot_offset: np.ndarray) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        self.p_nominal_by_leg[leg_name] = self.p_nominal_by_leg[leg_name] + np.asarray(foot_offset, dtype=float)
        self.p_des_by_leg[leg_name] = self.p_nominal_by_leg[leg_name].copy()

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


def run_viewer_foot_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Go2FootPositionController,
    print_tau_every: int,
    print_foot_every: int,
) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 135

        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            if print_tau_every > 0 and controller.step_count % print_tau_every == 0:
                print(f"t={data.time:.3f} tau: {controller.format_torque_map()}")
            if print_foot_every > 0 and controller.step_count % print_foot_every == 0:
                print(f"foot tracking t={data.time:.3f}")
                print(controller.format_foot_tracking())
            viewer.sync()


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
        "--lift-foot",
        choices=LEG_NAMES,
        help="Move one foot target up and down with a smooth cosine trajectory.",
    )
    parser.add_argument(
        "--lift-amplitude",
        type=float,
        default=0.02,
        help="Lift amplitude in meters for --lift-foot.",
    )
    parser.add_argument(
        "--lift-frequency",
        type=float,
        default=0.5,
        help="Lift frequency in Hz for --lift-foot.",
    )
    parser.add_argument(
        "--swing-foot",
        choices=LEG_NAMES,
        help="Move one foot target in x and z with a smooth periodic swing trajectory.",
    )
    parser.add_argument(
        "--swing-length",
        type=float,
        default=0.04,
        help="Peak-to-peak swing length in meters for --swing-foot.",
    )
    parser.add_argument(
        "--swing-diagonal",
        choices=sorted(DIAGONAL_PAIRS),
        help="Move a diagonal foot pair with the same x-z swing trajectory.",
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
    if args.lift_foot is not None:
        controller.configure_lift_motion(
            args.lift_foot,
            args.lift_amplitude,
            args.lift_frequency,
        )
    if args.swing_foot is not None:
        controller.configure_swing_motion(
            args.swing_foot,
            args.swing_length,
            args.lift_amplitude,
            args.lift_frequency,
        )
    if args.swing_diagonal is not None:
        controller.configure_diagonal_swing_motion(
            args.swing_diagonal,
            args.swing_length,
            args.lift_amplitude,
            args.lift_frequency,
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
        run_viewer_foot_control(
            model,
            data,
            controller,
            args.print_tau_every,
            args.print_foot_every,
        )


if __name__ == "__main__":
    main()
