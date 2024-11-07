from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.utils import busy_wait
from dataclasses import dataclass, field, replace
import time, random
import torch
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import JointState
from kortex_driver.msg import *
from gazebo_rl.environments.basic_arm import BasicArm
import numpy as np
from PIL import Image

from pynput import mouse

@dataclass
class RosRobotConfig:
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    state_topic: str = "/my_gen3/base_feedback"

global next_action
next_action = 0,0
minx, maxx = 500, 1500
miny, maxy = 100, 1200
def on_move(x, y):
    if x < minx or x > maxx or y < miny or y > maxy: return np.array([0, 0])
    
    xnorm = (x - minx) / (maxx - minx)
    ynorm = (y - miny) / (maxy - miny)

    xnorm = max(0, min(1, xnorm))
    ynorm = max(0, min(1, ynorm))

    xnorm = 0.2 * xnorm - .1
    ynorm = 0.2 * ynorm - .1

    # print(f'Pointer moved to {(x, y)} -> {xnorm, ynorm}')
    global next_action
    next_action = np.array([xnorm, ynorm])

import threading
current_observation = np.zeros(5)    
eef_lock = threading.Lock()
eef_time = time.time()
def eef_pose(data):
    global current_observation, eef_time
    # NOTE: ioda has many commented lines that should be referenced when adding state
    # TODO: this should just be a pose message.
    # augmented with velocity:
    x_pose = data.base.tool_pose_x 
    y_pose = data.base.tool_pose_y 
    z_pose = data.base.tool_pose_z

    with eef_lock:
        current_observation[0] = x_pose
        current_observation[1] = y_pose
        current_observation[2] = z_pose
        try:
            current_observation[3] = data.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        except:
            current_observation[3] = 0.0
            print("ERROR: No gripper feedback received.")
        current_observation[4] = 0.0 # REWARD
        
        dt = time.time() - eef_time
        if dt > 5: print(f"WARN: EEF time: {dt} seconds.")
        eef_time = time.time()

def sync_copy_eef():
    with eef_lock:
        return current_observation.copy()

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
        self.sim = rospy.get_param('/sim', True); print(f"Sim: {self.sim}")

        try:
            rospy.init_node("lerobot", anonymous=True, log_level=rospy.ERROR)
        except:
            print(f"Failed to start ros node!")
            raise RobotDeviceNotConnectedError

        rospy.Subscriber(self.config.state_topic, BaseCyclic_Feedback, callback=eef_pose)

        self.is_connected = False
        self.env = None
        self.listener = None
        self.robot_type = 'ros'

    def connect(self):
        self.env = BasicArm(sim=self.sim, 
                            action_duration=0.1,
                            velocity_control=True,
                            relative_commands=True)

        self.env.reset()
        # Connect the cameras
        for name in self.cameras:
            print(f"cam name", name)
            self.cameras[name].connect()
        self.is_connected = True

    def reset(self):
        print(f"RosRobot resetting...")
        self.env.reset()

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def init_teleop(self): 
        # Create a listener
        self.listener = mouse.Listener(on_move=on_move)
        self.listener.start()

    def run_calibration(self): 
        # TODO: Home arm or reset arm?
        # raise NotImplementedError
        pass

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )
        
        action = [0, 0, 0, 0, 0, 0, 0]
        if HARDCODE_ACTION:=True:
            obs_dict = self.capture_observation()
            print(f"WARN: hardcoded action")
            if obs_dict['observation.state'][0] < 0.8:action[0] = 0.05
            if obs_dict['observation.state'][1] < 0.2:action[1] = 0.05
        else:
            # Grab the teleoperation data and send it using ::send_action
            global next_action
            action = [next_action[0], next_action[1], 0, 0, 0, 0, 0]
            # TODO: Mouse control is fine for on robot verification, but then bring in xbox controller control.

        self.send_action(action)

        obs_dict = self.capture_observation()

        action_dict = {'action': torch.from_numpy(np.array(action))}

        # if not record_data: return
        # raise NotImplementedError
        return obs_dict, action_dict

    def capture_observation(self, display=False):
        before_eef_read_t = time.perf_counter()
        state = torch.from_numpy(sync_copy_eef())
        self.logs[f'eef_read'] = time.perf_counter() - before_eef_read_t

        # Output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state

        if self.sim:
            img = torch.randn((640, 480, 3)).float()
            obs_dict[f"observation.images.top"] = img
            
            if display:
                cv2.imshow('sim_image', obs_dict[f"observation.images.top"].numpy())
                cv2.waitKey(1)
        else:
            # Capture images from cameras
            images = {}
            for name in self.cameras:
                before_camread_t = time.perf_counter()
                images[name] = self.cameras[name].async_read()

                cv2.imshow(name, images[name]); cv2.waitKey(1);

                images[name] = torch.from_numpy(images[name])
                self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
                self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

            for name in self.cameras:
                obs_dict[f"observation.images.{name}"] = images[name]
                cv2.imshow(name, images[name])
                cv2.waitKey(1)

        # image_keys = [key for key in obs_dict if "image" in key]
        # for key in image_keys:
        #     cv2.imshow(key, obs_dict[key].numpy())
        # cv2.waitKey(1)
        return obs_dict


    def send_action(self, action):
        # Command the robot to take the action
        self.env.step(action)
        print(f"RosRobot::send_action {[f'{entry:1.2f}' for entry in action]}")
        return action

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ros is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False
        if self.listener: self.listener.stop()

if __name__ == '__main__':
    from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
    import traceback


    try:
        robot = RosRobot(cameras={
            "top": OpenCVCamera(0, fps=30, width=640, height=480)
        })
        robot.connect()
        while not rospy.is_shutdown():
            rospy.sleep(0.01)
            robot.teleop_step(record_data=False)
            try:
                # robot.capture_observation()
                robot.teleop_step(record_data=False)
            except Exception as e:
                print(f"Failed to capture observation: ", e)
    except Exception as e:
        print(f"Ros Loop died: ", e)
        print(traceback.format_exc())
    finally:
        robot.disconnect()

