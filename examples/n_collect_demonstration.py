from pathlib import Path
import sys, os

# NOTE: replace with path to peripheral controller
sys.path.append(os.path.expanduser("~/workspace/fastrl/nov20"))
from second_wind.peripheral import get_memorymaze_action_from_joystick, get_pusht_action_from_joystick, get_calvin_action_from_joystick, get_pinpad_action_from_joystick, get_ev3_action_from_joystick

from lerobot.common.envs.factory import make_env

env_name = "xarm"

# create an output directory for the demonstration
output_directory = Path(f"local/demonstration/{env_name}")
output_directory.mkdir(parents=True, exist_ok=True)


# create an environment
env = make_env({"env": {"name": env_name, "task": "XarmLift-v0"}})


