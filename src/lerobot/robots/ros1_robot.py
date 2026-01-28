import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import rospy
    import std_msgs.msg
    import sensor_msgs.msg
    import trajectory_msgs.msg
except ImportError:
    rospy = None

from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot


logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("ros1_robot")
@dataclass
class ROS1RobotConfig(RobotConfig):
    # Mapping from observation key to ROS topic
    observation_topics: dict[str, str] = field(default_factory=dict)
    # Mapping from action key to ROS topic
    action_topics: dict[str, str] = field(default_factory=dict)
    # Mapping from feature key to shape (tuple of ints). 
    # Used to define observation_features and action_features.
    # Keys should match keys in observation_topics and action_topics.
    # Example: {"state": (6,), "action": (6,)}
    features: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # List of joint names to extract from JointState message (if applicable)
    joint_names: list[str] | None = None
    # Whether to use JointTrajectory controller for actions
    use_joint_trajectory_controller: bool = False
    # ROS node name
    node_name: str = "lerobot_node"


class ROS1Robot(Robot):
    config_class = ROS1RobotConfig
    name = "ros1_robot"

    def __init__(self, config: ROS1RobotConfig):
        super().__init__(config)
        self.config = config
        self.subscribers = {}
        self.publishers = {}
        self._latest_observation = {}
        self.cameras = {}
        self._connected = False

    @property
    def observation_features(self) -> dict:
        features = {}
        for key, topic in self.config.observation_topics.items():
            if key in self.config.features:
                shape = self.config.features[key]
                if len(shape) == 1:
                    for i in range(shape[0]):
                        features[f"{key}_{i}"] = float
                elif len(shape) == 3:
                    features[key] = tuple(shape)
        return features

    @property
    def action_features(self) -> dict:
        features = {}
        for key, topic in self.config.action_topics.items():
            if key in self.config.features:
                shape = self.config.features[key]
                if len(shape) == 1:
                    for i in range(shape[0]):
                        features[f"{key}_{i}"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if rospy is None:
            raise ImportError("rospy is not installed. Please install ROS1 to use this robot.")
        
        if not self._connected:
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node(self.config.node_name, anonymous=True, disable_signals=True)
            except rospy.exceptions.ROSException:
                pass  # Node might already be initialized

            # Setup subscribers
            for key, topic in self.config.observation_topics.items():
                is_image = False
                if key in self.config.features:
                    shape = self.config.features[key]
                    if len(shape) == 3:
                        is_image = True
                
                if is_image:
                    if topic is None or topic.lower() == "dummy":
                        # Dummy camera support
                        logger.info(f"Configuring dummy camera for {key}")
                        shape = self.config.features[key]
                        # Just store a blank image immediately so it's ready
                        # Shape is (C, H, W) in config usually for LeRobot, but we store HWC internally for consistency if we decoded with opencv? 
                        # Wait, LeRobot usually expects CHW in the dataset but HWC from cameras?
                        # Let's match what _img_callback produces: HWC.
                        # Wait, config features usually (C, H, W).
                        # Let's assume config is (C, H, W).
                        # LeRobot/PyTorch usually handles HWC -> CHW conversion in the dataset or transform,
                        # BUT the config must matching (H, W, C) for LeRobot to interpret it correctly.
                        if len(shape) == 3:
                            h, w, c = shape
                            self._latest_observation[key] = np.zeros((h, w, c), dtype=np.uint8)
                            logger.info(f"Dummy camera '{key}' initialized with shape {self._latest_observation[key].shape} (HWC)")
                        else:
                            logger.warning(f"Shape {shape} for {key} is not length 3, treating as flat array.")
                            c = shape[0]
                            self._latest_observation[key] = np.zeros((c,), dtype=np.float32)
                    else:
                        self.subscribers[key] = rospy.Subscriber(
                            topic, 
                            sensor_msgs.msg.Image, 
                            lambda msg, k=key: self._img_callback(msg, k)
                        )
                elif self.config.joint_names and "state" in key:
                    # Heuristic: if joint_names are provided and key contains 'state', assume JointState
                    # This is a bit rigid, but works for the common case of /joint_states
                    self.subscribers[key] = rospy.Subscriber(
                        topic, 
                        sensor_msgs.msg.JointState, 
                        lambda msg, k=key: self._joint_state_callback(msg, k)
                    )
                else:
                    self.subscribers[key] = rospy.Subscriber(
                        topic, 
                        std_msgs.msg.Float32MultiArray, 
                        lambda msg, k=key: self._obs_callback(msg, k)
                    )

            # Setup publishers
            # Setup publishers
            for key, topic in self.config.action_topics.items():
                if self.config.use_joint_trajectory_controller and "action" in key:
                     self.publishers[key] = rospy.Publisher(
                        topic, 
                        trajectory_msgs.msg.JointTrajectory, 
                        queue_size=1
                    )
                else:
                     self.publishers[key] = rospy.Publisher(
                        topic, 
                        std_msgs.msg.Float32MultiArray, 
                        queue_size=1
                    )
            
            self._connected = True
            
            # Wait for first message on all topics
            logger.info("Waiting for first observation from ROS topics...")
            start_time = time.time()
            while time.time() - start_time < 10.0:
                all_received = True
                for key in self.config.observation_topics:
                    if key not in self._latest_observation:
                        all_received = False
                        break
                if all_received:
                    logger.info("Received observations from all topics.")
                    break
                time.sleep(0.1)
            else:
                logger.warning("Timed out waiting for observations from some topics. Robot might not be fully functional.")

    def _obs_callback(self, msg, key):
        # Store raw array
        self._latest_observation[key] = np.array(msg.data)

    def _img_callback(self, msg, key):
        # Decode ROS Image to numpy
        # Assuming rgb8 or similar 3-channel
        dtype = np.uint8
        channels = 3
        # Basic decoding for rgb8
        try:
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)
            self._latest_observation[key] = img
        except Exception as e:
            logger.error(f"Failed to decode image from {key}: {e}")

    def _joint_state_callback(self, msg, key):
        if not self.config.joint_names:
            return
            
        # Extract positions for the configured joint names
        positions = []
        try:
            # Create a lookup for faster access if needed, but linear scan is okay for small N
            # msg.name and msg.position are parallel
            name_to_pos = dict(zip(msg.name, msg.position))
            
            for name in self.config.joint_names:
                if name in name_to_pos:
                    positions.append(name_to_pos[name])
                else:
                    # Warn once? simple fallback
                    positions.append(0.0)
            
            self._latest_observation[key] = np.array(positions, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to parse JointState: {e}")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("Robot is not connected.")
        
        # Flatten observation for LeRobot compatibility
        obs = {}
        for key, value in self._latest_observation.items():
            if key in self.config.features:
                shape = self.config.features[key]
                if len(shape) == 1 and len(value) == shape[0]:
                    for i in range(shape[0]):
                        obs[f"{key}_{i}"] = float(value[i])
                elif len(shape) == 3:
                     obs[key] = value
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise RuntimeError("Robot is not connected.")
        
        # Reconstruct arrays from flat action dict
        # We need to group by key prefix
        arrays_to_send = {}
        for key, topic in self.config.action_topics.items():
            if key in self.config.features:
               shape = self.config.features[key]
               if len(shape) == 1:
                   values = []
                   for i in range(shape[0]):
                       val = action.get(f"{key}_{i}")
                       if val is not None:
                           values.append(val)
                   if len(values) == shape[0]:
                       arrays_to_send[key] = np.array(values, dtype=np.float32)

        for key, value in arrays_to_send.items():
            if key in self.publishers:
                if self.config.use_joint_trajectory_controller and "action" in key and self.config.joint_names:
                    msg = trajectory_msgs.msg.JointTrajectory()
                    msg.joint_names = self.config.joint_names
                    point = trajectory_msgs.msg.JointTrajectoryPoint()
                    point.positions = value.tolist()
                    point.time_from_start = rospy.Duration(0.01) # Small duration
                    msg.points = [point]
                    self.publishers[key].publish(msg)
                else:
                    msg = std_msgs.msg.Float32MultiArray()
                    msg.data = value.tolist()
                    self.publishers[key].publish(msg)
        
        return action

    def disconnect(self) -> None:
        for sub in self.subscribers.values():
            sub.unregister()
        for pub in self.publishers.values():
            pub.unregister()
        self._connected = False
