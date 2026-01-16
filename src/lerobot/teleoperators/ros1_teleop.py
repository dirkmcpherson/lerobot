
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import rospy
    import std_msgs.msg
    import sensor_msgs.msg
except ImportError:
    rospy = None

from lerobot.processor import RobotAction
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("ros1_teleop")
@dataclass
class ROS1TeleoperatorConfig(TeleoperatorConfig):
    # Mapping from action key to ROS topic
    action_topics: dict[str, str] = field(default_factory=dict)
    # Mapping from feedback key to ROS topic (optional)
    feedback_topics: dict[str, str] = field(default_factory=dict)
    # Mapping from feature key to shape (tuple of ints).
    features: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # ROS node name
    node_name: str = "lerobot_teleop_node"


class ROS1Teleoperator(Teleoperator):
    config_class = ROS1TeleoperatorConfig
    name = "ros1_teleop"

    def __init__(self, config: ROS1TeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.subscribers = {}
        self.publishers = {}
        self._latest_action = {}
        self._connected = False

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
    def feedback_features(self) -> dict:
        features = {}
        for key, topic in self.config.feedback_topics.items():
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
            raise ImportError("rospy is not installed. Please install ROS1 to use this teleoperator.")
        
        if not self._connected:
            try:
                if not rospy.core.is_initialized():
                    rospy.init_node(self.config.node_name, anonymous=True, disable_signals=True)
            except rospy.exceptions.ROSException:
                pass

            # Setup subscribers for actions (receiving actions from "teleop" device which is ROS)
            for key, topic in self.config.action_topics.items():
                self.subscribers[key] = rospy.Subscriber(
                    topic, 
                    std_msgs.msg.Float32MultiArray, 
                    lambda msg, k=key: self._action_callback(msg, k)
                )

            # Setup publishers for feedback
            for key, topic in self.config.feedback_topics.items():
                 self.publishers[key] = rospy.Publisher(
                    topic, 
                    std_msgs.msg.Float32MultiArray, 
                    queue_size=1
                )
            
            self._connected = True
            
            # Wait for first message on action topics
            logger.info("Waiting for first action from ROS topics...")
            start_time = time.time()
            while time.time() - start_time < 10.0:
                all_received = True
                for key in self.config.action_topics:
                    if key not in self._latest_action:
                        all_received = False
                        break
                if all_received:
                    logger.info("Received actions from all topics.")
                    break
                time.sleep(0.1)
            else:
                logger.warning("Timed out waiting for actions from some topics. Teleoperator might not be ready.")

    def _action_callback(self, msg, key):
        self._latest_action[key] = np.array(msg.data)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if not self._connected:
            raise RuntimeError("Teleoperator is not connected.")
        
        # Flatten action
        action = {}
        for key, value in self._latest_action.items():
            if key in self.config.features:
                shape = self.config.features[key]
                if len(shape) == 1 and len(value) == shape[0]:
                    for i in range(shape[0]):
                        action[f"{key}_{i}"] = float(value[i])
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self._connected:
            raise RuntimeError("Teleoperator is not connected.")
        
        # Flattened feedback processing would be similar, but for now assuming simple check
        # Assuming feedback might be flat or structured. 
        # Reconstruct arrays if needed.
        # This is reciprocal to send_action in Robot.
        arrays_to_send = {}
        for key, topic in self.config.feedback_topics.items():
            if key in self.config.features:
               shape = self.config.features[key]
               if len(shape) == 1:
                   values = []
                   for i in range(shape[0]):
                       val = feedback.get(f"{key}_{i}")
                       if val is not None:
                           values.append(val)
                   if len(values) == shape[0]:
                       arrays_to_send[key] = np.array(values, dtype=np.float32)

        for key, value in arrays_to_send.items():
            if key in self.publishers:
                msg = std_msgs.msg.Float32MultiArray()
                msg.data = value.tolist()
                self.publishers[key].publish(msg)

    def disconnect(self) -> None:
        for sub in self.subscribers.values():
            sub.unregister()
        for pub in self.publishers.values():
            pub.unregister()
        self._connected = False
