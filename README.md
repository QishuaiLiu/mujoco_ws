# MuJoCo Quadruped Starter

This is a fresh standalone quadruped controller project for MuJoCo.

It includes:

- a simple 12-DoF quadruped model
- a procedural foot trajectory generator
- a leg inverse-kinematics controller
- a MuJoCo simulation entrypoint

## Layout

- `models/quadruped.xml`: MuJoCo model
- `quadruped_mj/controller.py`: gait generator and controller
- `quadruped_mj/kinematics.py`: leg inverse kinematics
- `run_sim.py`: launch the simulation

## Install With Conda

```bash
conda env create -f environment.yml
conda activate mujoco-quad
```

If you use `mamba` or `micromamba`, the equivalent is:

```bash
mamba env create -f environment.yml
mamba activate mujoco-quad
```

or:

```bash
micromamba create -f environment.yml
micromamba activate mujoco-quad
```

## Install With `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_sim.py
```

Experimental trot:

```bash
python run_sim.py --trot
```

Controls:

- `Ctrl+C`: quit

The default behavior is:

1. reset into a stable standing pose
2. hold the standing pose

Use `--trot` to test the procedural trot. The trot is still experimental; the current stable baseline is standing.

## Notes

This is a starter controller, not a full whole-body controller.

It is intended to give you a clean base for the next steps:

- better state estimation
- contact-aware control
- MPC / whole-body control
- velocity command tracking
- terrain adaptation

## Next Good Steps

1. Add keyboard velocity commands.
2. Log body height, joint targets, and foot phases.
3. Replace the procedural gait with a state machine.
4. Add a simple stabilizing body PD loop.
