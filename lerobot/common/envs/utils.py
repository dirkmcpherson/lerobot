#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import einops
import numpy as np
import torch
from torch import Tensor
import cv2


def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    cv2.destroyAllWindows()

    if isinstance(observations["image"], dict):
        imgs = {f"observation.images.{key}": img for key, img in observations["image"].items()}
    else:
        imgs = {"observation.image": observations["image"]}

    for imgkey, img in imgs.items():
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

        img = torch.from_numpy(img)

        # sanity check that images are channel last
        if len(img.shape) == 3:
            img = img[np.newaxis, ...]
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)
        img /= 255

        return_observations[imgkey] = img

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    if 'vector_state' in observations:
        return_observations["observation.state"] = torch.from_numpy(observations["vector_state"]).float()
    else:
        return_observations["observation.state"] = torch.zeros(1)

    return return_observations
