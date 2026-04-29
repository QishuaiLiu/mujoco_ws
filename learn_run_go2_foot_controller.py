from __future__ import annotations

import argparse

import mujoco
import numpy as np

from quadruped_mj.go2_kinematics import GO2_LEGS
from run_go2 import GO2_SCENE, JOINT_NAMES, Go2StandController, run_headless, run_viewer

LEG_NAMES = ["FL", "FR", "RL", "RR"]
LEG_SLICES = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}

class Go2FootPositionController(Go2StandController):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        super().__init__(model, data)
        ## convert q_pos to p_pos
        self.p_des_by_leg = {leg_name: GO2_LEGS[leg_name].forward(self.q_des[LEG_SLICES[leg_name]]) for leg_name in LEG_NAMES}

    def step(self) -> None:
        self._update_joint_targets_from_feet()
        super().step()

    ## from current q pos to get the desired q pos and send it to controller to do a PD control
    def _update_joint_targets_from_feet(self) -> None:
        for leg_name in LEG_NAMES:
            leg_slice = LEG_SLICES[leg_name]
            q_current = self.data.qpos[self.qpos_ids[leg_slice]]
            self.q_des[leg_slice] = GO2_LEGS[leg_name].inverse(self.p_des_by_leg[leg_name], initial_guess=q_current,)


    def set_desired_foot_position(self, leg_name: str, foot_position: np.ndarray) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")
        self.p_des_by_leg[leg_name] = self.p_des_by_leg[leg_name] + np.asarray(foot_position, dtype=float)


    def offset_desired_foot_position(self, leg_name: str, foot_offset: np.ndarray) -> None:
        leg_name = leg_name.upper()
        if leg_name not in self.p_des_by_leg:
            raise KeyError(f"Unknown leg name: {leg_name}")

        self.p_des_by_leg[leg_name] = self.p_des_by_leg[leg_name] + np.asarray(foot_offset, dtype=float)

    def desired_foot_map(self) -> list[tuple[str, np.ndarray]]:
        return [(leg_name, self.p_des_by_leg[leg_name].copy()) for leg_name in LEG_NAMES]


def main() ->None:
    parser = argparse.ArgumentParser(description="Run the Go2 foot-position controller using IK + joint PD")
    parser.add_argument("--headless", action="store_true", help="Run without opening the viewer.")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless run duration in seconds.")
    parser.add_argument("--offset-foot", nargs=4, metavar=("LEG", "DX", "DY", "DZ"), help="Apply a base-frame foot offset in meters, for example: --offset-foot FL 0.01 0 0",)
    parser.add_argument("--print-tau-every", type=int, default=0, help="Print torque command every N control steps. 0 disables repeated print.")
    args = parser.parse_args()

    if not GO2_SCENE.exists():
        raise FileNotFoundError(f"Go2 scene not found: {GO2_SCENE}")

    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data = mujoco.MjData(model)
    controller = Go2FootPositionController(model, data)

    if args.offset_foot is not None:
        leg_name, dx, dy, dz = args.offset_foot
        controller.offset_desired_foot_position(leg_name, np.array([float(dx), float(dy), float(dz)], dtype=float),)
        controller.reset()


    print("Go2 desired foot target (p_des in base frame):")

    for leg_name, foot_position in controller.desired_foot_map():
        print(f" {leg_name}: {np.round(foot_position, 4)}")

    print("Go2 initial torque command (tau):")

    for joint_name, tau_value in controller.torque_map():
        print(f" {joint_name}: {tau_value:4f}")

    if args.headless:
        run_headless(model, data, controller, args.duration)
    else:
        run_viewer(model, data, controller, args.print_tau_every)

if __name__ == "__main__":
    main()

