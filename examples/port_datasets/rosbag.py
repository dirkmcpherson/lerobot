import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int

import importlib

def get_ros_msg_type(msg_type_str):
    """
    Dynamically import a ROS message type based on its string representation.
    :param msg_type_str: String representation of the ROS message type (e.g., "std_msgs/String").
    :return: ROS message class.
    """
    try:
        package_name, message_name = msg_type_str.split("/")
        print(f"Attempting to import {package_name}")
        module = importlib.import_module(f"{package_name}.msg")
        return getattr(module, message_name)
    except Exception as e:
        rospy.logerr(f"Failed to import message type '{msg_type_str}': {e}")
        raise

FPS=30
ROBOT_NAME="my_gen3_lite"
TASK = "Pick up a cup."
FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["x", "y", "z", "r", "p", "y", "gripper"],
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": {
            "axes": ["x", "y", "z", "r", "p", "y" "gripper"],
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
    },
    "observation.image.bottom": {
        "dtype": 'image',
        "shape": (3, 96, 96),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
}

import hydra
import rospy
from omegaconf import OmegaConf
from collections import deque
from functools import partial

def main(repo_id: str, push_to_hub: bool = True):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    robot_path = 'lerobot/configs/robot/ros.yaml'

    robot_cfg = init_hydra_config(robot_path)
    robot = hydra.utils.instantiate(robot_cfg)

    features = FEATURES
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=ROBOT_NAME,
        features=features,
        image_writer_threads=4,
    )

    def callback(msg, queue):
        print(f"{type(msg)}")
        queue.append(msg)

    subscriber_queue = {}
    for feature in FEATURES:
        if topic_info := OmegaConf.select(robot.topics, feature):
            print(feature, topic_info)
            msg_class = get_ros_msg_type(topic_info.msg_type)
            topic = topic_info.topic
            subscriber_queue[topic] = deque(maxlen=1)
            rospy.Subscriber(topic, msg_class, callback=partial(callback, queue=subscriber_queue[topic]), queue_size=1)
        # else:
        #     print(f"{feature} not in robot.topics")
    
    # from IPython import embed as ipshell; ipshell()
    rate = rospy.Rate(FPS)
    while not rospy.is_shutdown():
        # for k,v in subscriber_queue.items():
            # if len(v) > 0:
            #     print(k, len(v))
        frame = {}; valid = True
        for k, v in subscriber_queue.items():
            if len(v) == 0:
                valid = False
            else:
                frame[k] = v.pop()
        
        if valid: dataset.add_frame(frame)
        rate.sleep()


    # episodes = range(len(episode_data_index["from"]))
    # for ep_idx in episodes:
    #     from_idx = episode_data_index["from"][ep_idx]
    #     to_idx = episode_data_index["to"][ep_idx]
    #     num_frames = to_idx - from_idx

    #     for frame_idx in range(num_frames):
    #         i = from_idx + frame_idx
    #         frame = {
    #             "action": torch.from_numpy(action[i]),
    #             # Shift reward and success by +1 until the last item of the episode
    #             "next.reward": reward[i + (frame_idx < num_frames - 1)],
    #             "next.success": success[i + (frame_idx < num_frames - 1)],
    #         }

    #         frame["observation.state"] = torch.from_numpy(agent_pos[i])

    #         if mode == "keypoints":
    #             frame["observation.environment_state"] = torch.from_numpy(keypoints[i])
    #         else:
    #             frame["observation.image"] = torch.from_numpy(image[i])

    #         dataset.add_frame(frame)

    #     dataset.save_episode(task=TASK)

    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "js/ros_test"
    main(repo_id=repo_id, push_to_hub=False)

    # Uncomment if you want to load the local dataset and explore it
    # dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
    # breakpoint()
