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

Use `--trot` to test the procedural trot. The trot is conservative and open-loop: it should avoid the quick fall-down behavior, but some yaw drift is expected because there is not yet a heading controller.

## Run The Clean Unitree Go2 Model

The Go2 model is vendored from Google DeepMind's MuJoCo Menagerie under `third_party/mujoco_menagerie/unitree_go2`.

Headless load/control test:

```bash
python run_go2.py --headless --duration 5
```

Viewer:

```bash
python run_go2.py
```

This uses a simple joint-space PD torque controller around the Go2 `home` keyframe. It is only a standing test, not a walking controller.

## Run The Go2 Foot-Position Controller

This variant keeps desired foot positions `p_des` in the base frame, uses IK to compute joint targets `q_des`, and then applies the same joint-space PD torque control underneath.

Headless stand test:

```bash
python run_go2_foot_control.py --headless --duration 5
```

Try a small foot-space perturbation:

```bash
python run_go2_foot_control.py --headless --duration 5 --offset-foot FL 0.01 0 0
```

Try a simple scripted foot lift:

```bash
python run_go2_foot_control.py --headless --duration 2 --lift-foot FL --print-foot-every 100
```

Try a simple scripted foot swing:

```bash
python run_go2_foot_control.py --headless --duration 2 --swing-foot FL --print-foot-every 100
```

Try a diagonal pair swing:

```bash
python run_go2_foot_control.py --headless --duration 2 --swing-diagonal FL_RR --print-foot-every 100
```

Try alternating diagonal pairs:

```bash
python run_go2_foot_control.py --headless --duration 4 --alternate-diagonal --print-foot-every 500
```

The foot-control runner applies a small roll/pitch stabilization by default. To compare against the open-loop behavior:

```bash
python run_go2_foot_control.py --headless --duration 2 --swing-diagonal FL_RR --no-stabilize-body
```

## Verify Go2 Leg Kinematics

This project also includes a small Go2 leg kinematics module and a verification script that compares its forward kinematics against MuJoCo's home pose:

```bash
python verify_go2_kinematics.py
```

This is the next bridge from joint-space control to foot-space control.

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
