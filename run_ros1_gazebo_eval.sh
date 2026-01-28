#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source venv/bin/activate
source /home/j/catkin_ws/devel/setup.bash

echo "============================================="
echo "PHASE 3: DEPLOYMENT (Gazebo Evaluation)"
echo "============================================="

# Ensure cleanup
# ./cleanup.sh || true

echo "Launching Gazebo..."
roslaunch kortex_gazebo spawn_kortex_robot.launch start_gazebo:=true &
GAZEBO_PID=$!
echo "Waiting for Gazebo to start..."
sleep 15
echo "Unpausing Gazebo physics..."
rosservice call /gazebo/unpause_physics || true

# Check if data dir exists and clear it
# Check if data dir exists and clear it
if [ -d "data/eval_gazebo" ]; then
    echo "Removing existing data/eval_gazebo..."
    rm -rf data/eval_gazebo
fi

echo "Starting Policy Evaluation..."
# Run the policy
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [7], "action": [7]}' \
  --robot.joint_names='["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]' \
  --robot.use_joint_trajectory_controller=true \
  --robot.observation_topics='{"state": "/my_gen3/joint_states"}' \
  --robot.action_topics='{"action": "/my_gen3/gen3_joint_trajectory_controller/command"}' \
  --robot.id=kinova_sim \
  --robot.node_name=lerobot_eval_robot \
  --policy.path=outputs/train/gazebo_diffusion/checkpoints/last/pretrained_model \
  --dataset.repo_id=lerobot/eval_gazebo_data \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=8 \
  --dataset.reset_time_s=1 \
  --dataset.single_task="eval_reach_target" \
  --dataset.video=false \
  --display_data=false \
  --play_sounds=false \
  --dataset.push_to_hub=false \
  --dataset.root=data/eval_gazebo

echo "Evaluation Finished!"
kill $GAZEBO_PID || true
