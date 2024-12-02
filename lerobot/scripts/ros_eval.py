from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.robots.ros import RosRobot
import time 
import torch
from lerobot.common.robot_devices.utils import busy_wait

inference_time_s = 10
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/train/act_ros_test/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

robot = RosRobot()
robot.connect()

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        else:
            observation[name] = observation[name].type(torch.float32)

        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation


    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # Order the robot to move
    robot.send_action(action)

    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)