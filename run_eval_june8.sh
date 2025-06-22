#!/bin/bash


dataset_ids=("8_out" "8_out1" "8" "8_out" "8_out1" "8") # run through each twice
for datset_id in "${dataset_ids[@]}";
do
    python lerobot/scripts/eval.py -p outputs/train/dp_diffusion_${datset_id}/checkpoints/last/pretrained_model/ eval.n_episodes=50 eval.batch_size=1
done


