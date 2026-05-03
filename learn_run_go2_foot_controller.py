from __future__ import annotations

import argparse
import time

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
TROT_PHASE_OFFSETS = {
    "FL": 0.0,
    "RR": 0.0,
    "FR": 0.5,
    "RL": 0.5,
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
FOOT_GEOMS = {
    "FL": "FL",
    "FR": "FR",
    "RL": "RL",
    "RR": "RR",
}
FOOT_OFFSET = np.array([-0.002, 0.0, -0.213], dtype=float)
CONTACT_FORCE_THRESHOLD = 5.0

class Go2FootPositionController(Go2StandController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.foot_body_ids = {
            leg_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, FOOT_BODIES[leg_name])
            for leg_name in LEG_NAMES
        }
        self.foot_geom_ids = {
            leg_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOMS[leg_name]) for leg_name in LEG_NAMES
        }
        self.geom_id_to_leg = {geom_id: leg for leg, geom_id in self.foot_geom_ids.items()}
        ## convert q_pos to p_pos
        self.p_des_by_leg = {leg_name: GO2_LEGS[leg_name].forward(self.q_des[LEG_SLICES[leg_name]]) for leg_name in LEG_NAMES}
        self.p_nominal_by_leg = {
            leg_name: foot_position.copy() for leg_name, foot_position in self.p_des_by_leg.items()
        }

        self.lift_leg_name: str | None = None
        self.lift_amplitude = 0.0
        self.lift_frequency = 0.0
        self.swing_leg_name: str | None = None
        self.swing_length = 0.0
        self.swing_amplitude = 0.0
        self.swing_frequency = 0.0
        self.swing_diagonal_name: str | None = None
        self.alternate_diagonal_swing = False
        self.trot_forward_enabled = False
        self.trot_step_length = 0.0
        self.trot_step_height = 0.0
        self.trot_frequency = 0.0
        self.trot_swing_fraction = 0.5
        self.stabilize_body = False
        self.pitch_stabilization_gain = 0.08
        self.roll_stabilization_gain = 0.05

    def step(self) -> None:
        self._update_scripted_foot_targets()
        self._apply_body_stabilization()
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

    def configure_swing_motion(self, leg_name: str, length: float, amplitude: float, frequency: float,) -> None:
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

    def configure_body_stabilization(self, enabled: bool) -> None:
        self.stabilize_body = enabled

    def configure_alternating_diagonal_swing(
        self,
                length: float,
                amplitude: float,
                frequency: float,
    ) -> None:
        if length < 0.0:
            raise ValueError("Swing length must be non-negative")
        if amplitude < 0.0:
            raise ValueError("Swing amplitude must be non-negative")
        if frequency <= 0.0:
            raise ValueError("Swing frequency must be positive")

        self.alternate_diagonal_swing = True
        self.swing_length = float(length)
        self.swing_amplitude = float(amplitude)
        self.swing_frequency = float(frequency)

    def configure_forward_trot(
        self,
                step_length: float,
                step_height: float,
                frequency: float,
                swing_fraction: float = 0.5,
    ) -> None:
        if step_length < 0.0:
            raise ValueError("Step length must be non-negative")
        if step_height < 0.0:
            raise ValueError("Step height must be non-negative")
        if frequency <= 0.0:
            raise ValueError("Trot frequency must be positive")
        if not 0.0 < swing_fraction < 1.0:
            raise ValueError("Swing fraction must be between 0 and 1")
        self.trot_forward_enabled = True
        self.trot_step_length = float(step_length)
        self.trot_step_height = float(step_height)
        self.trot_frequency = float(frequency)
        self.trot_swing_fraction = float(swing_fraction)

    def _update_scripted_foot_targets(self) -> None:
        self._reset_desired_feet_to_nominal()
        if self.trot_forward_enabled:
            self._update_forward_trot_targets()
            return
        if self.lift_leg_name is None:
            pass
        else:
            phase = 2.0 * np.pi * self.lift_frequency * self.data.time
            z_lift = 0.5 * self.lift_amplitude * (1.0 - np.cos(phase))
            target = self.p_nominal_by_leg[self.lift_leg_name].copy()
            target[2] += z_lift
            self.p_des_by_leg[self.lift_leg_name] = target

        if (self.swing_leg_name is None and self.swing_diagonal_name is None and not self.alternate_diagonal_swing):
            return

        if self.alternate_diagonal_swing:
            cycle_phase = (self.swing_frequency * self.data.time) % 1.0
            if cycle_phase < 0.5:
                swing_legs = DIAGONAL_PAIRS["FL_RR"]
                local_phase = cycle_phase / 0.5
            else:
                swing_legs = DIAGONAL_PAIRS["FR_RL"]
                local_phase = (cycle_phase - 0.5) / 0.5
            x_swing = self.swing_length * (local_phase - 0.5)
            z_lift = self.swing_amplitude * np.sin(np.pi * local_phase)
            for leg_name in swing_legs:
                target = self.p_nominal_by_leg[leg_name].copy()
                target[0] += x_swing
                target[2] += z_lift
                self.p_des_by_leg[leg_name] = target
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

    ## from current q pos to get the desired q pos and send it to controller to do a PD control
    def _update_joint_targets_from_feet(self) -> None:
        for leg_name in LEG_NAMES:
            leg_slice = LEG_SLICES[leg_name]
            q_current = self.data.qpos[self.qpos_ids[leg_slice]]
            self.q_des[leg_slice] = GO2_LEGS[leg_name].inverse(self.p_des_by_leg[leg_name], initial_guess=q_current,)

    def _reset_desired_feet_to_nominal(self) -> None:
        for leg_name in LEG_NAMES:
            self.p_des_by_leg[leg_name] = self.p_nominal_by_leg[leg_name].copy()

    def _apply_body_stabilization(self) -> None:
        if not self.stabilize_body:
            return
        roll, pitch = self._base_roll_pitch()
        x_shift = np.clip(self.pitch_stabilization_gain * pitch, -0.025, 0.025)
        y_shift = np.clip(self.roll_stabilization_gain * roll, -0.02, 0.02)

        for leg_name in LEG_NAMES:
            self.p_des_by_leg[leg_name][0] += x_shift
            self.p_des_by_leg[leg_name][1] += y_shift

    def _update_forward_trot_targets(self) -> None:
        gait_phase = (self.trot_frequency * self.data.time) % 1.0
        for leg_name in LEG_NAMES:
            leg_phase = (gait_phase + TROT_PHASE_OFFSETS[leg_name]) % 1.0
            x_offset, z_offset = self._forward_trot_foot_offset(leg_phase)
            target = self.p_nominal_by_leg[leg_name].copy()
            target[0] += x_offset
            target[2] += z_offset
            self.p_des_by_leg[leg_name] = target

    def _forward_trot_foot_offset(self, leg_phase: float) -> tuple[float, float]:
        if leg_phase < self.trot_swing_fraction:
            swing_progress = leg_phase / self.trot_swing_fraction
            x_offset = self.trot_step_length * (swing_progress - 0.5)
            z_offset = self.trot_step_height * np.sin(np.pi * swing_progress)
            return float(x_offset), float(z_offset)

        stance_progress = (leg_phase - self.trot_swing_fraction) / (1.0 - self.trot_swing_fraction)
        x_offset = self.trot_step_length * (0.5 - stance_progress)
        return float(x_offset), 0.0

    def _base_roll_pitch(self) -> tuple[float, float]:
        w, x, y, z = self.data.qpos[3:7]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        return float(roll), float(pitch)

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
        return [(leg_name, self.p_des_by_leg[leg_name] - self.actual_foot_position(leg_name)) for leg_name in LEG_NAMES]

    def format_foot_tracking(self) -> str:
        lines = []
        for leg_name in LEG_NAMES:
            p_des = self.p_des_by_leg[leg_name]
            p_actual = self.actual_foot_position(leg_name)
            error = p_des - p_actual
            lines.append(
                f"{leg_name} p_des={np.round(p_des, 4)} "
                f"p_actual={np.round(p_actual, 4)} "
                f"err={np.round(error, 4)}"
            )
        return "\n".join(lines)

    def contact_state(self) -> dict[str, tuple[float, bool]]:
        force_buf = np.zeros(6, dtype=float)
        normal_force_by_leg = {leg_name: 0.0 for leg_name in LEG_NAMES}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            leg = self.geom_id_to_leg.get(int(contact.geom1)) or self.geom_id_to_leg.get(int(contact.geom2))
            if leg is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            normal_force_by_leg[leg] += float(force_buf[0])

        return {
            leg_name: (force, force > CONTACT_FORCE_THRESHOLD) for leg_name, force in normal_force_by_leg.items()
        }

    def format_contact_state(self) -> str:
        state = self.contact_state()
        parts = []
        for leg_name in LEG_NAMES:
            force, in_contact = state[leg_name]
            flag = "stance" if in_contact else "swing "
            parts.append(f"{leg_name} {flag} fn={force:6.2f}N")
        return "  ".join(parts)



def run_headless_foot_control(model: mujoco.MjModel,
                              data: mujoco.MjData,
                              controller: Go2FootPositionController,
                              duration: float,
                              print_foot_every: int,
                              print_contact_every: int,) -> None:
    steps = int(duration / model.opt.timestep)
    min_height = float("inf")
    start_xy = data.qpos[0:2].copy()

    for step in range(steps):
        controller.step()
        mujoco.mj_step(model, data)
        min_height = min(min_height, float(data.qpos[2]))

        if step % max(1, steps // 10) == 0:
            print(
                f"t={data.time:.2f} x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} z={data.qpos[2]:.3f} "
                f"quat={np.round(data.qpos[3:7], 3)}"
            )

        if print_foot_every > 0 and controller.step_count % print_foot_every == 0:
            print(f"foot tracking t={data.time:.3f}")
            print(controller.format_foot_tracking())

        if print_contact_every > 0 and controller.step_count % print_contact_every == 0:
            print(f"contact t={data.time:.3f} {controller.format_contact_state()}")

    displacement_xy = data.qpos[0:2] - start_xy
    print(
        f"done duration={data.time:.2f} "
        f"dx={displacement_xy[0]:.3f} dy={displacement_xy[1]:.3f} "
        f"final_z={data.qpos[2]:.3f} min_z={min_height:.3f}"
    )

def run_viewer_foot_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Go2FootPositionController,
    print_tau_every: int,
    print_foot_every: int,
    print_base_every: int,
    print_contact_every: int,
) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 135
        start_xy = data.qpos[0:2].copy()
        sim_start = data.time
        wall_start = time.perf_counter()

        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            if print_base_every > 0 and controller.step_count % print_base_every == 0:
                displacement_xy = data.qpos[0:2] - start_xy
                print(
                    f"base t={data.time:.3f} "
                    f"x={data.qpos[0]:.3f} y={data.qpos[1]:.3f} z={data.qpos[2]:.3f} "
                    f"dx={displacement_xy[0]:.3f} dy={displacement_xy[1]:.3f}"
                )

            if print_tau_every > 0 and controller.step_count % print_tau_every == 0:
                print(f"t={data.time:.3f} tau: {controller.format_torque_map()}")
            if print_foot_every > 0 and controller.step_count % print_foot_every == 0:
                print(f"foot tracking t={data.time:.3f}")
                print(controller.format_foot_tracking())
            if print_contact_every > 0 and controller.step_count % print_contact_every == 0:
                print(f"contact t={data.time:.3f} {controller.format_contact_state()}")
            viewer.sync()
            sleep_time = (data.time - sim_start) - (time.perf_counter() - wall_start)
            if sleep_time > 0.0:
                time.sleep(min(sleep_time, 0.01))

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Go2 foot-position controller using IK + joint PD")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless run duration in seconds.")
    parser.add_argument("--offset-foot", nargs=4, metavar=("LEG", "DX", "DY", "DZ"), help="Apply a base-frame foot offset in meters, for example: --offset-foot FL 0.01 0 0",)
    parser.add_argument("--print-tau-every", type=int, default=0, help="Print torque command every N control steps. 0 disables repeated print.")

    parser.add_argument("--print-foot-every", type=int, default=0, help="Print foot tracking error every N control steps. 0 disables repeated printing")

    parser.add_argument("--lift-foot", choices=LEG_NAMES, help="Move one foot target up and down with a smooth cosine trajectory.", )

    parser.add_argument("--lift-amplitude", type=float, default=0.02, help="Lift amplitude in meters for --lift-foot.", )

    parser.add_argument("--lift-frequency", type=float, default=0.5, help="Lift frequency in Hz for --lift-foot.",)

    parser.add_argument("--swing-foot", choices=LEG_NAMES, help="Move one foot target in x and z with a smooth periodic swing trajectory")

    parser.add_argument("--swing-length", type=float, default=0.04, help="Peak-to-peak swing length in meters for --swing-foot.")
    parser.add_argument("--swing-diagonal", choices=sorted(DIAGONAL_PAIRS), help="Move a diagonal foot pair with the same x-z swing trajectory.")

    parser.add_argument("--no-stabilize-body", action="store_true", help="Disable simple roll/pitch foot-placement stabilization.")

    parser.add_argument("--alternate-diagonal", action="store_true", help="Alternate the swing phase between FL+RR and FR+RL.")

    parser.add_argument("--trot-forward", action="store_true", help="Run an open-loop forward trot with swing and stance foot motion")

    parser.add_argument("--step-length", type=float, default=0.03, help="Peak-to-peak x motion in meters for --trot-forward")

    parser.add_argument("--step-height", type=float, default=0.015, help="Foot lift height in meters for --trot-forward")

    parser.add_argument("--gait-frequency", type=float, default=0.25, help="Gait cycle frequency in Hz for --trot-forward")

    parser.add_argument("--swing-fraction", type=float, default=0.5, help="Fraction of each gait cycle spent in swing for --trot-forward.")

    parser.add_argument("--print-contact-every", type=int, default=0, help="Print per-foot contact normal force every N control steps. 0 disables",)

    parser.add_argument(
        "--print-base-every",
        type=int,
        default=0,
        help="Print base position every N control steps in viewer mode. 0 disables repeated printing.",
    )

    args = parser.parse_args()

    if not GO2_SCENE.exists():
        raise FileNotFoundError(f"Go2 scene not found: {GO2_SCENE}")

    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data = mujoco.MjData(model)
    controller = Go2FootPositionController(model, data)
    controller.configure_body_stabilization(not args.no_stabilize_body)

    if args.offset_foot is not None:
        leg_name, dx, dy, dz = args.offset_foot
        controller.offset_desired_foot_position(leg_name, np.array([float(dx), float(dy), float(dz)], dtype=float),)

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
    if args.alternate_diagonal:
        controller.configure_alternating_diagonal_swing(
            args.swing_length,
            args.lift_amplitude,
            args.lift_frequency,
        )
    if args.trot_forward:
        controller.configure_forward_trot(args.step_length,
                                          args.step_height,
                                          args.gait_frequency,
                                          args.swing_fraction)

    controller.reset()

    print("Go2 actual foot position (p_actual in base frame):")
    for leg_name, foot_position in controller.actual_foot_map():
        print(f" {leg_name}: {np.round(foot_position, 4)}")


    print("Go2 desired foot target (p_des in base frame):")

    for leg_name, foot_position in controller.desired_foot_map():
        print(f" {leg_name}: {np.round(foot_position, 4)}")

    print("Go2 foot tracking error (p_des - p_actual): ")
    for leg_name, error in controller.foot_error_map():
        print(f"{leg_name}: {np.round(error, 4)}")

    print("Go2 initial torque command (tau):")

    for joint_name, tau_value in controller.torque_map():
        print(f" {joint_name}: {tau_value:.4f}")

    if args.headless:
        run_headless_foot_control(model, data, controller, args.duration, args.print_foot_every, args.print_contact_every,)
    else:
        run_viewer_foot_control(model, data, controller, args.print_tau_every, args.print_foot_every, args.print_base_every, args.print_contact_every,)

if __name__ == "__main__":
    main()
