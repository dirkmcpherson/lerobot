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
python -u scripts/ros1_mock_env.py > mock_eval_env.log 2>&1 &
MOCK_PID=$!
sleep 3

# Cleanup previous eval data
if [ -d "data/eval" ]; then
    echo "Removing previous eval data directory..."
    rm -rf data/eval
fi

echo "Starting Policy Evaluation..."
# Run the policy on the robot
# We use lerobot_record to run the policy and record the result
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [3], "action": [3], "laptop": [96, 96, 3]}' \
  --robot.observation_topics='{"state": "/mock_obs", "laptop": "/mock_image"}' \
  --robot.action_topics='{"action": "/mock_action"}' \
  --robot.id=ros_tester \
  --robot.node_name=lerobot_eval_robot \
  --policy.path=outputs/train/ros1_test/checkpoints/last/pretrained_model \
  --dataset.repo_id=lerobot/ros1_eval_data \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=5.0 \
  --dataset.reset_time_s=0.5 \
  --dataset.single_task="eval_ros1" \
  --dataset.video=true \
  --display_data=false \
  --dataset.push_to_hub=false \
  --dataset.root=data/eval

echo "Evaluation finished!"

kill $MOCK_PID
