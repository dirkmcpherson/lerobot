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
# TODO(rcadene, alexander-soare): clean this file
"""Borrowed from https://github.com/fyhMer/fowm/blob/main/src/logger.py"""

import logging
import os
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import OmegaConf
from termcolor import colored

from lerobot.common.policies.policy_protocol import Policy

# local logging reqs
from torch.utils.tensorboard import SummaryWriter
import imageio.v3 as iio
import numpy as np

def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def cfg_to_group(cfg, return_list=False):
    """Return a group name for logging. Optionally returns group name as list."""
    lst = [
        f"policy:{cfg.policy.name}",
        f"dataset:{cfg.dataset_repo_id}",
        f"env:{cfg.env.name}",
        f"seed:{cfg.seed}",
    ]
    return lst if return_list else "-".join(lst)


class Logger:
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, job_name, cfg):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._job_name = job_name
        self._model_dir = self._log_dir / "checkpoints"
        self._buffer_dir = self._log_dir / "buffers"
        self._save_model = cfg.training.save_model
        self._disable_wandb_artifact = cfg.wandb.disable_artifact
        self._save_buffer = cfg.training.get("save_buffer", False)
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._cfg = cfg
        self._eval = []
        project = cfg.get("wandb", {}).get("project")
        entity = cfg.get("wandb", {}).get("entity")
        enable_wandb = cfg.get("wandb", {}).get("enable", False)
        run_offline = not enable_wandb or not project
        if run_offline:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
            self._wandb = None
        else:
            os.environ["WANDB_SILENT"] = "true"
            import wandb

            wandb.init(
                project=project,
                entity=entity,
                name=job_name,
                notes=cfg.get("wandb", {}).get("notes"),
                # group=self._group,
                tags=cfg_to_group(cfg, return_list=True),
                dir=self._log_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                # TODO(rcadene): add resume option
                resume=None,
            )
            print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
            logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
            self._wandb = wandb
        self._tensorboard = None
        if self._cfg.tensorboard.enable:
            self._tensorboard = SummaryWriter(log_dir=self._log_dir)
            logging.info(colored("Tensorboard logs will be saved locally.", "yellow", attrs=["bold"]))

    def save_model(self, policy: Policy, identifier):
        if self._save_model:
            self._model_dir.mkdir(parents=True, exist_ok=True)
            save_dir = self._model_dir / str(identifier)
            policy.save_pretrained(save_dir)
            # Also save the full Hydra config for the env configuration.
            OmegaConf.save(self._cfg, save_dir / "config.yaml")
            if self._wandb and not self._disable_wandb_artifact:
                # note wandb artifact does not accept ":" or "/" in its name
                artifact = self._wandb.Artifact(
                    f"{self._group.replace(':', '_').replace('/', '_')}-{self._seed}-{identifier}",
                    type="model",
                )
                artifact.add_file(save_dir / SAFETENSORS_SINGLE_FILE)
                self._wandb.log_artifact(artifact)

    def save_buffer(self, buffer, identifier):
        self._buffer_dir.mkdir(parents=True, exist_ok=True)
        fp = self._buffer_dir / f"{str(identifier)}.pkl"
        buffer.save(fp)
        if self._wandb and not self._disable_wandb_artifact:
            # note wandb artifact does not accept ":" or "/" in its name
            artifact = self._wandb.Artifact(
                f"{self._group.replace(':', '_').replace('/', '_')}-{self._seed}-{identifier}",
                type="buffer",
            )
            artifact.add_file(fp)
            self._wandb.log_artifact(artifact)

    def finish(self, agent, buffer):
        if self._save_model:
            self.save_model(agent, identifier="final")
        if self._save_buffer:
            self.save_buffer(buffer, identifier="buffer")
        if self._wandb:
            self._wandb.finish()

    def log_dict(self, d, step, mode="train"):
        assert mode in {"train", "eval"}
        if self._wandb is not None:
            for k, v in d.items():
                if not isinstance(v, (int, float, str)):
                    logging.warning(
                        f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
                    )
                    continue
                self._wandb.log({f"{mode}/{k}": v}, step=step)
        if self._tensorboard is not None:
            for k, v in d.items():
                self._tensorboard.add_scalar(f"{mode}/{k}", v, global_step=step)
            self._tensorboard.flush()

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        assert mode in {"train", "eval"}
        if self._wandb is not None:
            wandb_video = self._wandb.Video(video_path, fps=self._cfg.fps, format="mp4")
            self._wandb.log({f"{mode}/video": wandb_video}, step=step)
        if self._tensorboard is not None:
            # Read video file and convert it to tensorboard format
            frames = iio.imread(video_path, plugin="pyav")
            video_np = np.array(list(frames))
            T, H, W, C = video_np.shape
            # Transpose the channel position and add a leading 1 for the batch dimension expected by TF
            video_np = video_np.transpose(0, 3, 1, 2).reshape(1, T, C, H, W)
            self._tensorboard.add_video(f"{mode}/video", video_np, step, fps=self._cfg.fps)
