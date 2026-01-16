#!/bin/bash
set -e
source venv/bin/activate

# Check for roscore
if ! pgrep -x "roscore" > /dev/null; then
    echo "Error: roscore is not running."
    echo "Please open a NEW TERMINAL and run: 'roscore'"
    exit 1
fi

echo "Starting Mock ROS1 Environment..."
# Run unbuffered so we see logs
python -u scripts/ros1_mock_env.py > mock_env.log 2>&1 &
MOCK_PID=$!
sleep 3

# Cleanup previous data
if [ -d "data" ]; then
    echo "Removing previous data directory..."
    rm -rf data
fi

echo "Starting Recording..."
# Record 4 episodes (approx 4 trajectories)
# Robot subscribes to observation.
# Teleop subscribes to action.
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [3], "action": [3], "laptop": [96, 96, 3]}' \
  --robot.observation_topics='{"state": "/mock_obs", "laptop": "/mock_image"}' \
  --robot.action_topics='{"action": "/mock_action"}' \
  --robot.id=ros_tester \
  --robot.node_name=lerobot_recorder_robot \
  --teleop.type=ros1_teleop \
  --teleop.features='{"action": [3]}' \
  --teleop.action_topics='{"action": "/mock_action"}' \
  --teleop.id=ros_supervisor \
  --teleop.node_name=lerobot_recorder_teleop \
  --dataset.repo_id=lerobot/ros1_test_data \
  --dataset.num_episodes=4 \
  --dataset.episode_time_s=1.5 \
  --dataset.reset_time_s=0.5 \
  --dataset.single_task="test_ros1" \
  --dataset.video=true \
  --display_data=false \
  --dataset.push_to_hub=false \
  --dataset.root=data

echo "Recording finished. Starting Training..."

# Train a small diffusion policy
# Reduce batch size and steps for quick test
python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=lerobot/ros1_test_data \
  --dataset.root=data \
  --policy.type=diffusion \
  --output_dir=outputs/train/ros1_test \
  --job_name=ros1_test \
  --policy.device=cuda \
  --policy.repo_id=lerobot/ros1_test_policy \
  --policy.push_to_hub=false \
  --steps=500 \
  --save_freq=500 \
  --eval_freq=0 \
  --batch_size=4 \
  --wandb.enable=false

echo "Training finished successfully!"

kill $MOCK_PID
