#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source venv/bin/activate
# Source the user's workspace to find the kortex packages
source /home/j/catkin_ws/devel/setup.bash

echo "Starting Kinova Gen 3 Gazebo Simulation..."
# Launch in background
roslaunch kortex_gazebo spawn_kortex_robot.launch start_gazebo:=true &
GAZEBO_PID=$!

echo "Waiting for Gazebo to start..."
sleep 15

# Cleanup previous data
if [ -d "data/gazebo_test" ]; then
    echo "Removing previous data directory..."
    rm -rf data/gazebo_test
fi

echo "Starting Dummy Action Publisher..."
python scripts/ros1_dummy_action.py > dummy_action.log 2>&1 &
DUMMY_PID=$!

echo "Starting Recording from Gazebo..."
# Record 2 episodes
# Robot subscribes to joint states and publishes trajectory commands
# We use namespace 'my_gen3' as defined in spawn_kortex_robot.launch defaults
python src/lerobot/scripts/lerobot_record.py \
  --robot.type=ros1_robot \
  --robot.features='{"state": [7], "action": [7]}' \
  --robot.joint_names='["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]' \
  --robot.use_joint_trajectory_controller=true \
  --robot.observation_topics='{"state": "/my_gen3/joint_states"}' \
  --robot.action_topics='{"action": "/my_gen3/gen3_joint_trajectory_controller/command"}' \
  --robot.id=kinova_sim \
  --robot.node_name=lerobot_recorder_robot \
  --teleop.type=ros1_teleop \
  --teleop.features='{"action": [7]}' \
  --teleop.action_topics='{"action": "/dummy_action"}' \
  --teleop.id=dummy_teleop \
  --teleop.node_name=lerobot_recorder_teleop \
  --dataset.repo_id=lerobot/gazebo_test_data \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=5 \
  --dataset.reset_time_s=1 \
  --dataset.single_task="test_gazebo" \
  --dataset.video=false \
  --display_data=false \
  --dataset.push_to_hub=false \
  --dataset.root=data/gazebo_test

echo "Recording finished!"

kill $GAZEBO_PID
kill $DUMMY_PID
