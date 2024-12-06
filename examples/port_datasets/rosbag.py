import shutil
from pathlib import Path

import numpy as np
import torch

import std_msgs

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.ros import RosRobot
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say, none_or_int

import importlib

import sys
import cv2
import std_msgs.msg

def get_ros_msg_type(msg_type_str):
    """
    Dynamically import a ROS message type based on its string representation.
    :param msg_type_str: String representation of the ROS message type (e.g., "std_msgs/String").
    :return: ROS message class.
    """
    try:
        package_name, message_name = msg_type_str.split("/")
        print(f"\tAttempting to import {package_name}")
        module = importlib.import_module(f"{package_name}.msg")
        return getattr(module, message_name)
    except Exception as e:
        rospy.logerr(f"Failed to import message type '{msg_type_str}': {e}")
        raise

FPS=30
ROBOT_NAME="my_gen3_lite"
TASK = "Pick up a cup."
FEATURES = RosRobot.get_features()

import hydra
import rospy
from omegaconf import OmegaConf
from collections import deque
from functools import partial
import os

global EP_COMPLETE
EP_COMPLETE = False
def ep_complete_cb(msg: std_msgs.msg.Bool):
    global EP_COMPLETE
    EP_COMPLETE = msg.data

# misc helper functions
def check_dir(base_dir, uid):
    return not os.path.isdir(get_filename(base_dir, uid))
def get_filename(base_dir, uid):
  return os.path.join(base_dir, 'ros_') + str(uid).zfill(3)

def get_next_available_user_ind(base_dir, lo=-1, hi=float('inf')):
    # this is waaaaaay too fancy and bug-prone
    # but i wanted to implement binary search
    # and you can't stop me
    # TODO: anything but this
    assert hi > lo
    if lo < 0:
        if check_dir(base_dir, 0):
            return 0
        else:
            return get_next_available_user_ind(base_dir, lo=0, hi=hi)
    elif hi == float('inf'):
        q = lo+1
        while not check_dir(base_dir, q):
            q = q*2
        return get_next_available_user_ind(base_dir, lo=lo, hi=q)
    elif hi == lo+1:
        return hi
    else:
        q = int( (lo + hi) / 2)
        if check_dir(base_dir, q):
            return get_next_available_user_ind(base_dir, lo=lo, hi=q)
        else:
            return get_next_available_user_ind(base_dir, lo=q, hi=hi)

def main(repo_id: str, push_to_hub: bool = True):
    # if (LEROBOT_HOME / repo_id).exists():
    #     print(f"Deleting old repo")
    #     shutil.rmtree(LEROBOT_HOME / repo_id)
    next_uid = get_next_available_user_ind(repo_id)
    repo_id = get_filename(repo_id, next_uid)

    rospy.Subscriber('/playback_complete', std_msgs.msg.Bool, callback=ep_complete_cb)


    robot_path = 'lerobot/configs/robot/ros.yaml'
    robot_cfg = init_hydra_config(robot_path)
    robot = hydra.utils.instantiate(robot_cfg)

    print(os.listdir(LEROBOT_HOME))

    features = FEATURES
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=repo_id,
        robot_type=ROBOT_NAME,
        features=features,
        image_writer_threads=4,
    )

    def callback(msg, queue, feature):
        data = RosRobot.convert_rosmsg_to_target_type(msg, feature)
        # check types / size / format
        f = RosRobot.get_features()[feature]
        # dtype, shape = f['dtype'], f['shape']
        # assert type(f) == np.dtype(dtype)
        wall_ts = rospy.Time.now()
        queue.append((data, wall_ts))

    import signal
    global FINISHED
    FINISHED = False
    def handle_sigint(signal_number, frame):
        print("\nCtrl+C detected! Exiting gracefully.")
        global FINISHED
        FINISHED = True

    # Set the custom signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    ## Outer loop 
    rate = rospy.Rate(FPS); stale_msg_threshold = 0.1 # no messages older than this 
    global EP_COMPLETE
    break_loop = False
    EP_COMPLETE = False # just to start us off # NOTE: this flag is used to get us in and out of loops, so it's only aptly named during operation
    while not FINISHED:
        print(f"Starting bag capture.")
        ## Single bag (one bag per participant)
        n_frames = 0
        subscriber_queues = {}
        topic_to_feature_map = {}
        for feature in FEATURES:
            if topic_info := OmegaConf.select(robot.topics, feature):
                print(f'adding ', feature, topic_info)
                msg_class = get_ros_msg_type(topic_info.msg_type)
                topic = topic_info.topic
                subscriber_queues[topic] = deque(maxlen=1)
                topic_to_feature_map[topic] = feature
                rospy.Subscriber(topic, msg_class, callback=partial(callback, queue=subscriber_queues[topic], feature=feature), queue_size=1)
        ##

        action_topic = [k for k, v in topic_to_feature_map.items() if v == 'action'][0]; print(f"{action_topic=}")
        while not EP_COMPLETE and not FINISHED:
            frame = {}; valid = True; dropped_messages = 0
            wall_time = rospy.Time.now()
            # NOTE: We align with action frames, because they are the limiting factor in teleop data. This is another extremely specific implementation that needs to be abstracted.
            if all([len(v) > 0 for k,v in subscriber_queues.items()]):
                # TODO: we _must_ check for stale timestamps if we're doing it this 
                for k,v in subscriber_queues.items():
                    data, timestamp = v.pop()
                    if (wall_time - timestamp).to_sec() > stale_msg_threshold:
                        valid = False; break
                    else: 
                        frame[topic_to_feature_map[k]] = data
            else: continue

            if valid: 
                # print(f'adding valid frame')
                n_frames += 1
                if n_frames % 30 == 0: print(f"Added frame {n_frames}. {dropped_messages=}")
                dataset.add_frame(frame)
            else:
                dropped_messages += len(frame)
                # if dropped_messages > 10 and dropped_messages % 100 == 0:
                #     print(f"WARN: {dropped_messages=}")
            rate.sleep()

        if not n_frames:
            dataset.save_episode(TASK, encode_videos=False)
            
        print(f"Finished episode. {dropped_messages=}")
        EP_COMPLETE = False

    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "js"
    main(repo_id=repo_id, push_to_hub=False)

    # Uncomment if you want to load the local dataset and explore it
    # dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
    # breakpoint()
