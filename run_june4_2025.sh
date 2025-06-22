#!/bin/bash
# This script is used to run the training of the diffusion policy on the genesis dataset.

# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/genesis_8_out_None/ policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion_8out hydra.job.name=dp_diffusion_8out device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1

# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/genesis_8_out_None/ policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion_8out hydra.job.name=dp_diffusion_8out device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1



export LEROBOT_HOME=''

# Define the dataset ID strings to run
# dataset_ids=("8_out" "8_out1" "8")
dataset_ids=("8")

for datset_id in "${dataset_ids[@]}";
do
    python lerobot/scripts/train.py dataset_repo_id=local/genesis_${datset_id}_None/ policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion_${datset_id} hydra.job.name=dp_diffusion_${datset_id} device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1
done


# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/genesis_8_out_None/ policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion_8out hydra.job.name=dp_diffusion_8out device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1