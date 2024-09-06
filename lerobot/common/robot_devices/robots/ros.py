from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.cameras.utils import Camera
from dataclasses import dataclass, field, replace
import torch
import rospy

@dataclass
class RosRobotConfig:
        cameras: dict[str, Camera] = field(default_factory=lambda: {})


class RosRobot(Robot):
    def __init__(self,
                 config = None,
                 **kwargs) -> None:
        super().__init__()
        if config is None:
            config = RosRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

    def init_teleop(self): 
        raise NotImplementedError

    def run_calibration(self): 
        # TODO: Home arm or reset arm?
        raise NotImplementedError

    def teleop_step(self, record_data=False):

        # Grab the teleoperation data and send it using ::send_action
        action = None

        if not record_data: return

        obs_dict = self.capture_observation()

        action_dict = {'action': torch.from_numpy(action)}

        raise NotImplementedError

    def capture_observation(self): 
        raise NotImplementedError
        # Populate output dictionnaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = torch.from_numpy(state)
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
        return obs_dict


    def send_action(self, action):
        raise NotImplementedError
        success = False
        # Command the robot to take the action
        return success

if __name__ == '__main__':
    from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
    robot = RosRobot(cameras={
        "top": OpenCVCamera(0, fps=30, width=640, height=480)
    })