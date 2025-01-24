from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.utils import busy_wait
from dataclasses import dataclass, field, replace
import time, random
import torch
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image as RosImage, Joy
from kortex_driver.msg import *
from std_msgs.msg import Float32, Int8
from gazebo_rl.environments.basic_arm import BasicArm
from cv_bridge import CvBridge
import numpy as np
# from PIL import Image

from pynput import mouse

@dataclass
class RosRobotConfig:
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    topics: dict[str, str] = field(default_factory=lambda: {})
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
current_observation = np.zeros(13, dtype=np.float32)    
eef_lock = threading.Lock()
eef_time = time.time()
def eef_pose(data):
    global current_observation, eef_time

    with eef_lock:
        current_observation = RosRobot.basecyclicfeedback_to_state(data)
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
        self.topics = self.config.topics
        self.is_connected = False
        self.logs = {}
        self.sim = rospy.get_param('/sim', False); print(f"Sim: {self.sim}")

        try:
            rospy.init_node("lerobot", anonymous=True, log_level=rospy.ERROR)
        except:
            print(f"Failed to start ros node!")
            raise RobotDeviceNotConnectedError

        self.robot_name = rospy.get_param('robot_name', 'my_gen3_lite')

        print(f"Subscribing to state topic: {self.config.state_topic}")
        rospy.Subscriber('/my_gen3_lite/base_feedback', BaseCyclic_Feedback, callback=eef_pose)
        # rospy.Subscriber('/reward', std_msgs.msg.Float32, callback=self.CB)

        # make a top and bottom video publisher for visualization
        self.bridge = CvBridge()
        self.pub_top = rospy.Publisher('/top_image', RosImage, queue_size=10)
        self.pub_bottom = rospy.Publisher('/bottom_image', RosImage, queue_size=10)


        self.is_connected = False
        self.env = None
        self.listener = None
        self.robot_type = 'ros'

        self.crop_dim = 700
        self.crop_left_offset = 200

    def CB(self, data):
        print(f"Received data: {data}")

    def connect(self):
        ## TODO: The arm is a node and you're sending Twist messages on the 'robot_control' topic
        ## subscribe to the camera topic
        self.env = BasicArm(robot_name=self.robot_name,
                            sim=self.sim, 
                            action_duration=0.1,
                            velocity_control=True,
                            relative_commands=True,
                            max_action=0.11,
                            min_action=-0.11)

        # self.env.reset()
        # Connect the cameras # NOTE: needs to be moved. this is for native lerobot data collection from the arm (gen3 lite)
        for name in self.cameras:
            print(f"Connecting to camera", name)
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
        rospy.sleep(0.01)
        before_eef_read_t = time.perf_counter()
        state = torch.from_numpy(sync_copy_eef())
        self.logs[f'eef_read'] = time.perf_counter() - before_eef_read_t

        # print(f"lerobotros::capture_observation: {state}")

        # Output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state

        if self.sim:
            img = torch.randn((640, 480, 3)).float()
            obs_dict[f"observation.image.top"] = img
            
            if display:
                cv2.imshow('sim_image', obs_dict[f"observation.image.top"].numpy())
                cv2.waitKey(1)
        else:
            # Capture images from cameras
            images = {}
            for name in self.cameras:
                before_camread_t = time.perf_counter()

                img = self.cameras[name].async_read()
                if self.crop_dim > 0:
                    # crop from the right edge for TOP image
                    if name == 'top':
                        img = img[:self.crop_dim, self.crop_left_offset:self.crop_dim+self.crop_left_offset]
                        img = cv2.resize(img, (96, 96))
                        self.pub_top.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
                    else:
                        # crop from the left, no offset for BOTTOM image
                        img = img[:self.crop_dim, -self.crop_dim:]
                        img = cv2.resize(img, (96, 96))
                        self.pub_bottom.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))

                # images[name] = cv2.resize(img, (96, 96))

                images[name] = torch.from_numpy(img)

                # cv2.imshow(f'{name} {img.shape}', img); cv2.waitKey(1)

                self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
                self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

            for name in self.cameras:
                obs_dict[f"observation.image.{name}"] = images[name]

        return obs_dict


    def send_action(self, action):
        # Command the robot to take the action
        self.env.step(action)
        print(f"RosRobot::send_action {[f'{entry:+1.2f}' for entry in action]}")
        if action[-1] < -0.5:
            print(f"Closing gripper")
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

    @property
    def camera_features(self):
        return {k:v for k,v in RosRobot.get_features().items() if 'image' in k}
    
    @property
    def motor_features(self):
        return {k:v for k,v in RosRobot.get_features().items() if 'state' in k or 'action' in k}

    @staticmethod
    def basecyclicfeedback_to_state(msg: BaseCyclic_Feedback):
        gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
        tool_pose = msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z 
        tool_v = msg.base.tool_twist_linear_x, msg.base.tool_twist_linear_y, msg.base.tool_twist_linear_z, msg.base.tool_twist_angular_x, msg.base.tool_twist_angular_y, msg.base.tool_twist_angular_z
        return np.array([*tool_pose, *tool_v, gripper_pos], dtype=np.float32)

    @classmethod
    # NOTE: This is ugly because the behavior here can deviate from what rosbag.py descripb
    def convert_rosmsg_to_target_type(cls, msg, feature):
        # from IPython import embed as ipshell; ipshell()
        if type(msg) == RosImage:
            if not hasattr(cls, 'cvbridge'): cls.cvbridge = CvBridge()
            cvimg = cls.cvbridge.imgmsg_to_cv2(msg)
            # f = cls.get_features()[feature]
            # dtype, shape = f['dtype'], f['shape']
            cvimg_resized = np.array(cv2.resize(cvimg, (96, 96))) # TODO: match config shape
            cvimg_resized = np.transpose(cvimg_resized, axes=[2, 0, 1])
            # print(cvimg.shape, cvimg_resized.shape)
            
            # save out the image
            # cv2.imwrite(f'/home/j/workspace/{feature}.png', cvimg_resized)

            return cvimg_resized
        elif type(msg) == BaseCyclic_Feedback:
            state = RosRobot.basecyclicfeedback_to_state(msg)
            return state
        elif type(msg) == Joy: 
            # NOTE: unfortunately this depends on the input device.

            # FOR XBOX CONTROLLER
            if len(msg.buttons) >= 6:
                gripper_vel = -msg.buttons[4] if msg.buttons[4] else msg.buttons[5]
                x, y = msg.axes[0], msg.axes[1] # NOTE: this is wrong actually, and should be switched. for RSS we're manually switching the dimensions in the robot control loop
                z = msg.axes[4]
                r, p, yaw = 0., 0., msg.axes[3]
                return np.array([x, y, z, r, p, yaw, gripper_vel], dtype=np.float32)
            else:
                # FOR MOUSE & KEYBOARD
                gripper_state = 1.0 if msg.buttons[0] else 0 #TODO: make continuous and align with gripper direction
                gripper_state = -1.0 if msg.buttons[1] else 0
                return np.array([*msg.axes, gripper_state], dtype=np.float32)


        elif type(msg) in [Float32, Int8]:
            return np.float32(msg.data)
        elif type(msg) == std_msgs.msg.Bool:
            return np.float32(msg.data)
        # elif isinstance(msg, ):
        #     pass
        else:
            raise ValueError(type(msg)) #, msg)

    @classmethod 
    # NOTE: this should be configurable 
    def get_features(cls):
        if not hasattr(cls, 'features'):
            cls.features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (13,),
                    "names": {
                        "axes": ["x", "y", "z", "r", "p", "y", "vx", "vy", "vz", "vr", "vp", "vy", "gripper"],
                    },
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": {
                        "axes": ["vx", "vy", "vz", "vr", "vp", "vy" "vgripper"],
                    },
                },
                "next.reward": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": None,
                },
                "next.success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": None,
                },
                # "observation.environment_state": {
                #     "dtype": "float32",
                #     "shape": (16,),
                #     "names": [
                #         "keypoints",
                #     ],
                # },
                "observation.image.top": {
                    "dtype": 'image',
                    "shape": (3, 96, 96),
                    "names": [
                        "channel",
                        "height",
                        "width",
                    ],
                    'fps': 30
                },
                "observation.image.bottom": {
                    "dtype": 'image',
                    "shape": (3, 96, 96),
                    "names": [
                        "channel",
                        "height",
                        "width",
                    ],
                    'fps': 30
                },
            }
        return cls.features


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

