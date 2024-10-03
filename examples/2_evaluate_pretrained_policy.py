"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from datasets import Dataset, Features, Image, Sequence, Value
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.scripts.push_dataset_to_hub import save_meta_data

import cv2
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


def to_hf_dataset(data_dict, video):
    features = {}

    if video:
        features["observation.image"] = VideoFrame()
    else:
        features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    # TODO(rcadene): add success
    # features["next.success"] = Value(dtype='bool', id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset

env_name = "pusht"; img_size = 96; neps = 200

# Create a directory to store the video of the evaluation
# output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory = Path(f"~/workspace/lerobot/local/{env_name}/example_pusht_diffusion_{img_size}").expanduser()
output_directory.mkdir(parents=True, exist_ok=True)

# Download the diffusion policy for pusht environment
pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Device set to:", device)
else:
    device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.diffusion.num_inference_steps = 10

policy.to(device)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
    observation_width=img_size,
    observation_height=img_size
)
fps=10; video=False; video_path=None
ep_dicts = []
episode_data_index = {"from": [], "to": []}
ep_from = 0; ep_to = 0
for episode_index in range(neps):
    # Reset the policy and environmens to prepare for rollout
    policy.reset()
    numpy_observation, info = env.reset(seed=np.random.randint(0, 1000000))

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())

    step = 0
    done = False
    ep_dict = {}
    for key in ['observation.image', 'observation.state', 'action', 'episode_index', 'next.reward', 'next.done', 'frame_index', 'timestamp']:
        ep_dict[key] = []

    while not done:
        np_img = numpy_observation["pixels"]
        np_pos = numpy_observation["agent_pos"] if "agent_pos" in numpy_observation else "state"
        np_img = cv2.resize(np_img, (0,0), fx=4.0, fy=4.0) 
        np_img = cv2.putText(np_img, 'x'.join([f'{entry:03.1f}' for entry in np_pos]), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(np_pos)
        image = torch.from_numpy(numpy_observation["pixels"])

        # Convert to float32 with image from channel first in [0,255]
        # to channel last in [0,1]
        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)



        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        # Create the policy input dictionary
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            action = policy.select_action(observation)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        np_img = cv2.putText(np_img, 'x'.join([f'{entry:03.1f}' for entry in numpy_action]), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('eval', np_img)
        cv2.waitKey(1)

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done

        # if step > 10:
        #     print(f"Quitting early for debugging")
        #     done = True

        
        ep_dict['observation.image'].append(numpy_observation['pixels'])
        ep_dict['observation.state'].append(numpy_observation['agent_pos'])
        ep_dict['action'].append(numpy_action)
        ep_dict['episode_index'].append(episode_index)
        ep_dict['next.reward'].append(reward)
        ep_dict['next.done'].append(done)
        ep_dict['frame_index'].append(step)
        ep_dict['timestamp'].append(step / fps)

        step += 1; ep_to += 1
    
    # convert things to torch
    for key in ep_dict:
        if key == 'episode_index':
            ep_dict[key] = torch.tensor(ep_dict[key], dtype=torch.int64)
        elif key == 'observation.image':
            pass # keep it as a list of numpy arrays
        else:
            ep_dict[key] = torch.tensor(ep_dict[key])

    ep_dicts.append(ep_dict)
    episode_data_index["from"].append(ep_from); episode_data_index["to"].append(ep_to)
    ep_from = ep_to

if len(ep_dicts) == 0:
    print("No terminal step found in the dataset")
else:
    # from IPython import embed; embed()
    data_dict = concatenate_episodes(ep_dicts)

    for k,v in data_dict.items():
        print(k, v.shape if hasattr(v, 'shape') else len(v))

    hf_dataset = to_hf_dataset(data_dict, video)

    info = {"fps": fps, "video": video}

    if video_path: 
        print(f"video path: {video_path}")
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=env_name,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=video_path,
        )


    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(output_directory / "train"))
print(lerobot_dataset, 'written to ', output_directory)
stats = compute_stats(lerobot_dataset, batch_size=16, num_workers=1)
save_meta_data(info, stats, episode_data_index, output_directory / "meta_data")
print('metadata written to ', output_directory / "meta_data")

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
