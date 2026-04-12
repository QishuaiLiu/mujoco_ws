from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from quadruped_mj.kinematics import LegGeometry, leg_ik


@dataclass(frozen=True)
class LegConfig:
    name: str
    is_left: bool
    hip_x: float
    hip_y: float
    phase: float


class QuadrupedController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.dt = model.opt.timestep
        self.time = 0.0

        self.legs = [
            LegConfig("fl", True, 0.24, 0.11, 0.0),
            LegConfig("fr", False, 0.24, -0.11, 0.5),
            LegConfig("rl", True, -0.24, 0.11, 0.5),
            LegConfig("rr", False, -0.24, -0.11, 0.0),
        ]
        self.geometry = LegGeometry(hip_offset_y=0.0, upper_leg=0.22, lower_leg=0.24)

        self.joint_ids = {
            leg.name: np.array(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg.name}_hip_abd"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg.name}_hip_pitch"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg.name}_knee"),
                ],
                dtype=int,
            )
            for leg in self.legs
        }
        self.actuator_ids = {
            leg.name: np.array(
                [
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg.name}_hip_abd_pos"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg.name}_hip_pitch_pos"),
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{leg.name}_knee_pos"),
                ],
                dtype=int,
            )
            for leg in self.legs
        }

        self.stand_height = -0.42
        self.stance_half_length = 0.02
        self.gait_period = 0.55
        self.step_height = 0.07
        self.step_length = 0.12
        self.startup_duration = 3.0
        self.enable_trot = False

    def reset_pose(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.44
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        targets = self._compute_joint_targets(0.0)
        self._set_joint_positions(targets)
        self._apply_joint_targets(targets)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        self.time += self.dt
        targets = self._compute_joint_targets(self.time)
        self._apply_joint_targets(targets)

    def _compute_joint_targets(self, time_now: float) -> dict[str, np.ndarray]:
        if self.enable_trot:
            speed_scale = np.clip((time_now - self.startup_duration) / 1.0, 0.0, 1.0)
        else:
            speed_scale = 0.0
        targets: dict[str, np.ndarray] = {}

        for leg in self.legs:
            foot = self._desired_foot_position(leg, time_now, speed_scale)
            targets[leg.name] = leg_ik(foot, leg.is_left, self.geometry)

        return targets

    def _desired_foot_position(
        self,
        leg: LegConfig,
        time_now: float,
        speed_scale: float,
    ) -> np.ndarray:
        if speed_scale <= 0.0:
            phase = 0.0
        else:
            phase = ((time_now / self.gait_period) + leg.phase) % 1.0

        x = leg.hip_x
        y = leg.hip_y
        z = self.stand_height

        if speed_scale <= 0.0:
            return np.array([0.0, y, z], dtype=float)

        if phase < 0.5:
            stance_phase = phase / 0.5
            x_offset = (0.5 - stance_phase) * self.step_length
            z_offset = 0.0
        else:
            swing_phase = (phase - 0.5) / 0.5
            x_offset = (swing_phase - 0.5) * self.step_length
            z_offset = self.step_height * np.sin(np.pi * swing_phase)

        return np.array([x_offset, y, z + speed_scale * z_offset], dtype=float)

    def _apply_joint_targets(self, targets: dict[str, np.ndarray]) -> None:
        for leg in self.legs:
            self.data.ctrl[self.actuator_ids[leg.name]] = targets[leg.name]

    def _set_joint_positions(self, targets: dict[str, np.ndarray]) -> None:
        for leg in self.legs:
            for joint_id, value in zip(self.joint_ids[leg.name], targets[leg.name]):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = value
