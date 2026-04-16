from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=float,
    )


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class Go2LegKinematics:
    name: str
    hip_offset: np.ndarray
    thigh_offset: np.ndarray
    calf_offset: np.ndarray
    foot_offset: np.ndarray
    initial_guess: np.ndarray

    def forward(self, q: np.ndarray) -> np.ndarray:
        """Foot position in base frame for joint angles [hip_abd, hip_pitch, knee]."""
        q = np.asarray(q, dtype=float)
        r_hip = rot_x(q[0])
        r_thigh = r_hip @ rot_y(q[1])
        r_calf = r_thigh @ rot_y(q[2])
        return (
            self.hip_offset
            + r_hip @ self.thigh_offset
            + r_thigh @ self.calf_offset
            + r_calf @ self.foot_offset
        )

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        base = self.forward(q)
        jac = np.zeros((3, 3), dtype=float)
        for i in range(3):
            q_perturbed = q.copy()
            q_perturbed[i] += eps
            jac[:, i] = (self.forward(q_perturbed) - base) / eps
        return jac

    def inverse(
        self,
        foot_position: np.ndarray,
        initial_guess: np.ndarray | None = None,
        max_iters: int = 80,
        damping: float = 1e-4,
        tolerance: float = 1e-6,
    ) -> np.ndarray:
        """Damped least-squares IK for the 3-DoF Go2 leg."""
        target = np.asarray(foot_position, dtype=float)
        q = np.asarray(
            self.initial_guess if initial_guess is None else initial_guess,
            dtype=float,
        ).copy()

        for _ in range(max_iters):
            error = target - self.forward(q)
            if np.linalg.norm(error) < tolerance:
                return q
            jac = self.numerical_jacobian(q)
            lhs = jac.T @ jac + damping * np.eye(3)
            rhs = jac.T @ error
            q += np.linalg.solve(lhs, rhs)

        return q


GO2_LEGS: dict[str, Go2LegKinematics] = {
    "FL": Go2LegKinematics(
        name="FL",
        hip_offset=np.array([0.1934, 0.0465, 0.0]),
        thigh_offset=np.array([0.0, 0.0955, 0.0]),
        calf_offset=np.array([0.0, 0.0, -0.213]),
        foot_offset=np.array([-0.002, 0.0, -0.213]),
        initial_guess=np.array([0.0, 0.9, -1.8]),
    ),
    "FR": Go2LegKinematics(
        name="FR",
        hip_offset=np.array([0.1934, -0.0465, 0.0]),
        thigh_offset=np.array([0.0, -0.0955, 0.0]),
        calf_offset=np.array([0.0, 0.0, -0.213]),
        foot_offset=np.array([-0.002, 0.0, -0.213]),
        initial_guess=np.array([0.0, 0.9, -1.8]),
    ),
    "RL": Go2LegKinematics(
        name="RL",
        hip_offset=np.array([-0.1934, 0.0465, 0.0]),
        thigh_offset=np.array([0.0, 0.0955, 0.0]),
        calf_offset=np.array([0.0, 0.0, -0.213]),
        foot_offset=np.array([-0.002, 0.0, -0.213]),
        initial_guess=np.array([0.0, 0.9, -1.8]),
    ),
    "RR": Go2LegKinematics(
        name="RR",
        hip_offset=np.array([-0.1934, -0.0465, 0.0]),
        thigh_offset=np.array([0.0, -0.0955, 0.0]),
        calf_offset=np.array([0.0, 0.0, -0.213]),
        foot_offset=np.array([-0.002, 0.0, -0.213]),
        initial_guess=np.array([0.0, 0.9, -1.8]),
    ),
}
