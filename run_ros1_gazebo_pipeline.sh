#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source venv/bin/activate
source /home/j/catkin_ws/devel/setup.bash

# --- 1. Data Collection ---
echo "============================================="
echo "PHASE 1: DATA COLLECTION (Gazebo + Expert)"
echo "============================================="

# Start Gazebo
# roslaunch kortex_gazebo spawn_kortex_robot.launch start_gazebo:=true &
# GAZEBO_PID=$!
# Wait for Gazebo
# sleep 15

# Instead of managing Gazebo background process inside the script which is brittle,
# we assume Gazebo is running OR we launch it and trap exit.
# Let's try launching it.
echo "Launching Gazebo..."
roslaunch kortex_gazebo spawn_kortex_robot.launch start_gazebo:=true &
GAZEBO_PID=$!
echo "Waiting for Gazebo to start..."
sleep 15
echo "Unpausing Gazebo physics..."
rosservice call /gazebo/unpause_physics || true

# Check if data dir exists and clear it
if [ -d "data/gazebo_expert" ]; then
    rm -rf data/gazebo_expert
fi

echo "Starting Expert Demonstrator..."
python scripts/ros1_gazebo_expert.py > expert.log 2>&1 &
EXPERT_PID=$!

echo "Starting Recording..."
# Record 5 episodes
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [7], "action": [7]}' \
  --robot.joint_names='["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]' \
  --robot.use_joint_trajectory_controller=true \
  --robot.observation_topics='{"state": "/my_gen3/joint_states"}' \
  --robot.action_topics='{"action": "/my_gen3/gen3_joint_trajectory_controller/command"}' \
  --robot.id=kinova_sim \
  --robot.node_name=lerobot_recorder_robot \
  --robot.features='{"state": [7], "action": [7], "images": [96, 96, 3]}' \
  --robot.observation_topics='{"state": "/my_gen3/joint_states", "images": "dummy"}' \
  --teleop.type=ros1_teleop \
  --teleop.features='{"action": [7]}' \
  --teleop.action_topics='{"action": "/expert_action"}' \
  --teleop.id=expert_teleop \
  --teleop.node_name=lerobot_recorder_teleop \
  --dataset.repo_id=lerobot/gazebo_expert_data \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=8 \
  --dataset.reset_time_s=2 \
  --dataset.single_task="reach_target" \
  --dataset.video=true \
  --display_data=false \
  --dataset.push_to_hub=false \
  --dataset.root=data/gazebo_expert

echo "Data Collection Finished."
kill $EXPERT_PID || true

# --- 2. Training ---
echo "============================================="
echo "PHASE 2: TRAINING (Diffusion Policy)"
echo "============================================="

# Clean output dir
if [ -d "outputs/train/gazebo_diffusion" ]; then
    rm -rf outputs/train/gazebo_diffusion
fi

python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=lerobot/gazebo_expert_data \
  --dataset.root=data/gazebo_expert \
  --policy.type=diffusion \
  --output_dir=outputs/train/gazebo_diffusion \
  --job_name=gazebo_diffusion \
  --policy.device=cuda \
  --policy.repo_id=lerobot/gazebo_policy \
  --policy.push_to_hub=false \
  --steps=500 \
  --save_freq=500 \
  --log_freq=100 \
  --eval_freq=-1 

echo "Training Finished."

# --- 3. Deployment / Evaluation ---
echo "============================================="
echo "PHASE 3: DEPLOYMENT (Gazebo)"
echo "============================================="

# Ensure Gazebo is still running (it should be)
if [ -d "data/eval_gazebo" ]; then
    rm -rf data/eval_gazebo
fi

echo "Starting Policy Evaluation..."
# Run the policy
# Note: episode_time_s should match task duration
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [7], "action": [7]}' \
  --robot.joint_names='["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]' \
  --robot.use_joint_trajectory_controller=true \
  --robot.observation_topics='{"state": "/my_gen3/joint_states"}' \
  --robot.action_topics='{"action": "/my_gen3/gen3_joint_trajectory_controller/command"}' \
  --robot.id=kinova_sim \
  --robot.node_name=lerobot_eval_robot \
  --robot.features='{"state": [7], "action": [7], "images": [3, 96, 96]}' \
  --robot.observation_topics='{"state": "/my_gen3/joint_states", "images": "dummy"}' \
  --policy.path=outputs/train/gazebo_diffusion/checkpoints/last/pretrained_model \
  --dataset.repo_id=lerobot/eval_gazebo_data \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=8 \
  --dataset.reset_time_s=1 \
  --dataset.single_task="eval_reach_target" \
  --dataset.video=true \
  --display_data=false \
  --play_sounds=false \
  --dataset.push_to_hub=false \
  --dataset.root=data/eval_gazebo

echo "Pipeline Finished!"
kill $GAZEBO_PID
