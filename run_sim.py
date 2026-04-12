from pathlib import Path
import argparse

import mujoco
import mujoco.viewer

from quadruped_mj.controller import QuadrupedController


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MuJoCo quadruped simulation.")
    parser.add_argument(
        "--trot",
        action="store_true",
        help="Enable the experimental procedural trot after startup.",
    )
    args = parser.parse_args()

    model_path = Path(__file__).resolve().parent / "models" / "quadruped.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    controller = QuadrupedController(model, data)
    controller.enable_trot = args.trot
    controller.reset_pose()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120

        while viewer.is_running():
            controller.step()
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
