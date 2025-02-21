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
from sensor_msgs.msg import Image as RosImage, Joy, JointState


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
gripper_state = np.zeros((1,), dtype=np.float32)    
eef_lock = threading.Lock()
eef_time = time.time()
def eef_pose(data):
    global gripper_state, eef_time

    with eef_lock:
        gripper_state = RosRobot.basecyclicfeedback_to_state(data)
        dt = time.time() - eef_time
        if dt > 5: print(f"WARN: EEF time: {dt} seconds.")
        eef_time = time.time()

def sync_copy_eef():
    with eef_lock:
        return gripper_state.copy()
    
joint_state = np.zeros((6,), dtype=np.float32)
joint_lock = threading.Lock()
def state_from_jointstate(msg):
    with joint_lock:
        joint_state[:] = msg.position[:6]

def sync_copy_jointstate():
    with joint_lock:
        return joint_state.copy()
    
current_joy = Joy()
joy_lock = threading.Lock()
def on_joy(data):
    global current_joy
    with joy_lock:
        current_joy = data

def sync_copy_joy():
    with joy_lock:
        joy_copy = Joy()
        joy_copy.axes = current_joy.axes
        joy_copy.buttons = current_joy.buttons
        return joy_copy

class RosRobot(Robot):
    def __init__(self,
                 config = None,
                 **kwargs) -> None:
        super().__init__()
        if config is None:
            config = RosRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.cameras = self.config.cameras if self.config.cameras else {}
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
        rospy.Subscriber('/joy', Joy, callback=on_joy)
        rospy.Subscriber('/my_gen3_lite/joint_states', JointState, callback=state_from_jointstate)
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

        # initialize the pygame joystick
        # pygame.init()
        # pygame.joystick.init()
        # self.joystick = pygame.joystick.Joystick(0)
        # self.joystick.init()

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
            try:
                self.cameras[name].connect()
            except Exception as e:
                print(f"Failed to connect to camera {name}: ", e)

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
        
        joy = sync_copy_joy()

        CLOSE_BUTTON = 5; OPEN_BUTTON = 4
        if joy.buttons:
            if joy.buttons[CLOSE_BUTTON]: action = np.array([0, 0, 0, 0, -1])
            elif joy.buttons[OPEN_BUTTON]: action = np.array([0, 0, 0, 0, 1])
            else: action = np.array([joy.axes[1], joy.axes[0], joy.axes[4], joy.axes[3], 0])
        else: action = np.array([0, 0, 0, 0, 0])

            # vx, vy, vz, vr, vp, vyaw = cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z, cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z
            # if abs(vx) > VELOCITY_CAP: vx = VELOCITY_CAP if vx > 0 else -VELOCITY_CAP
            # if abs(vy) > VELOCITY_CAP: vy = VELOCITY_CAP if vy > 0 else -VELOCITY_CAP
            # if abs(vz) > VELOCITY_CAP: vz = VELOCITY_CAP if vz > 0 else -VELOCITY_CAP

            # if self._constraints and current_state:
            #     newx = current_state.base.tool_pose_x + vx * self.buffer_multiplier * self.CARTESIAN_VELOCITY_DURATION
            #     newy = current_state.base.tool_pose_y + vy * self.buffer_multiplier * self.CARTESIAN_VELOCITY_DURATION
            #     newz = current_state.base.tool_pose_z + vz * self.buffer_multiplier * self.CARTESIAN_VELOCITY_DURATION
            #     if (newx < self.minx and vx < 0) or (newx > self.maxx and vx > 0): vx = 0; print(f"WARN: constraining zeroing x velocity. {newx=:1.2f}")
            #     if (newy < self.miny and vy < 0) or (newy > self.maxy and vy > 0): vy = 0; print(f"WARN: constraining zeroing y velocity. {newy=:1.2f}")
            #     if (newz < self.minz and vz < 0) or (newz > self.maxz and vz > 0): print(f"WARN: constraining zeroing z velocity. {newz=:1.2f} {vz=:1.2f}"); vz = 0; 


            # cmd = [vx,vy,vz,vr,vp,vyaw]
            # print([f'{entry:+1.1f}' for entry in cmd], f' DT: {time.time() - self.t0:1.2f}   t: {time.time():1.2f}'); self.t0 = time.time()
            # self._robot.cartesian_velocity_command(cmd, duration=self.CARTESIAN_VELOCITY_DURATION, block=False, radians=True)
            # print(f"Command ended.")

        self.send_action(action)
        obs_dict = self.capture_observation()
        action_dict = {'action': torch.from_numpy(np.array(action))}

        # if not record_data: return
        # raise NotImplementedError
        return obs_dict, action_dict

    def capture_observation(self, display=False):
        rospy.sleep(0.01)
        before_eef_read_t = time.perf_counter()
        self.logs[f'eef_read'] = time.perf_counter() - before_eef_read_t

        # print(f"lerobotros::capture_observation: {state}")
        eef_state = torch.from_numpy(sync_copy_eef())
        joint_state = torch.from_numpy(sync_copy_jointstate())
        state = torch.cat([joint_state, eef_state], dim=0)

        # Output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        print('State: ', ','.join([f'{entry:+1.2f}' for entry in state]))
        obs_dict['observation.environment_state'] = torch.from_numpy(np.zeros((2,), dtype=np.float32))

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

                if self.crop_dim > 0:
                    # crop from the right edge for TOP image
                    if name == 'top':
                        img = self.cameras[name].async_read()
                        img = img[:self.crop_dim, self.crop_left_offset:self.crop_dim+self.crop_left_offset]
                        img = cv2.resize(img, (96, 96))
                        self.pub_top.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
                    else:
                        # img = self.cameras[name].async_read()
                        # # crop from the left, no offset for BOTTOM image
                        # img = img[:self.crop_dim, -self.crop_dim:]
                        # img = cv2.resize(img, (96, 96))
                        # self.pub_bottom.publish(self.bridge.cv2_to_imgmsg(img, encoding="bgr8"))
                        img = np.zeros((96, 96, 3), dtype=np.uint8)
                        self.cameras[name].logs["delta_timestamp_s"] = 0.0

                # images[name] = cv2.resize(img, (96, 96))

                # convert the image to a 3 channel grayscale
                # cv2.imwrite(f'/home/j/workspace/{name}_cropped.png', img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # cv2.imwrite(f'/home/j/workspace/{name}_gray.png', img)
                img = np.stack([img]*3, axis=-1)
                # cv2.imwrite(f'/home/j/workspace/{name}_3ch.png', img)
                ##

                # print(f"in capture_observation: {name} {img.max()} {img.shape}")

                images[name] = torch.from_numpy(img)

                # cv2.imshow(f'{name} {img.shape}', img); cv2.waitKey(0)
                # write out the image
                # cv2.imwrite(f'/home/j/workspace/{name}.png', img)

                self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
                self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

            for name in self.cameras:
                obs_dict[f"observation.image.{name}"] = images[name]

        return obs_dict


    def send_action(self, action):
        # Command the robot to take the action
        self.env.step(action)
        print(f"RosRobot::send_action {[f'{entry:+1.2f}' for entry in action]}")
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
        # return np.array([*tool_pose, *tool_v, gripper_pos], dtype=np.float32)
        # return np.array([*tool_pose[:3], gripper_pos], dtype=np.float32)
        return np.array([gripper_pos], dtype=np.float32)


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
            print(cvimg_resized.shape)

            # if the image only has one channel, we need to add two more
            if cvimg_resized.shape[0] == 1:
                cvimg_resized = np.concatenate([cvimg_resized, cvimg_resized, cvimg_resized], axis=0)
            
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
                    # "shape": (13,),
                    "shape": (7,),
                    "names": {
                        "axes": ["x", "y", "z", "r", "p", "yaw", "gripper"],
                    },
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": {
                        # "axes": ["vx", "vy", "vz", "vyaw", "vgripper"],
                        "axes": ["x", "y", "z", "r", "p", "yaw", "gripper"],
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
                "observation.environment_state": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": [
                        "filler",
                    ],
                },
                # "observation.image.top": {
                #     "dtype": 'image',
                #     "shape": (3, 96, 96),
                #     "names": [
                #         "channel",
                #         "height",
                #         "width",
                #     ],
                #     'fps': 30
                # },
                # "observation.image.bottom": {
                #     "dtype": 'image',
                #     "shape": (3, 96, 96),
                #     "names": [
                #         "channel",
                #         "height",
                #         "width",
                #     ],
                #     'fps': 30
                # },
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

