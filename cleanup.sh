#!/bin/bash
echo "Killing ROS and Gazebo processes..."
pkill -9 -f gazebo
pkill -9 -f roslaunch
pkill -9 -f rosmaster
pkill -9 -f roscore
pkill -9 -f rosout
# Kill specific python scripts we started
pkill -9 -f "python scripts/ros1_gazebo_expert.py"
pkill -9 -f "python src/lerobot/scripts/lerobot_record.py"
pkill -9 -f "python src/lerobot/scripts/lerobot_train.py"
pkill -9 -f "python scripts/ros1_dummy_action.py"
# Kill the pipeline script itself if running
pkill -9 -f "run_ros1_gazebo_pipeline.sh"
echo "Cleanup complete."
