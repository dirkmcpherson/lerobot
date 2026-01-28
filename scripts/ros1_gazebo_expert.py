#!/usr/bin/env python3
import rospy
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState

class ExpertDemonstrator:
    def __init__(self):
        rospy.init_node('expert_demonstrator', anonymous=True)
        
        # Publisher for the 'teleop' action (to be recorded)
        self.action_pub = rospy.Publisher('/expert_action', Float32MultiArray, queue_size=10)
        
        # Subscribe to robot state to know where we are
        self.current_joints = np.zeros(7)
        self.sub = rospy.Subscriber('/my_gen3/joint_states', JointState, self.state_callback)
        
        # Kinova Gen 3 Joint Limits (approximate, usually +/- 2pi or unlimited for some)
        # We'll just define a safe "Home" and "Target"
        # Home: All zeros (except elbow maybe?)
        self.home_pose = np.array([0.0, -0.35, 3.14, -2.54, 0.0, -0.87, 1.57]) # Similar to spawn
        # Target: Move joint 1 (Base) to 180 deg and lift joint 2 (Shoulder)
        self.target_pose = np.array([3.14, 0.5, 3.14, -1.5, 0.0, -0.87, 1.57])
        
        self.rate = rospy.Rate(10) # 10Hz control loop
        self.logger = rospy.loginfo

    def state_callback(self, msg):
        # Map joint names to 7-dof array
        # Names: joint_1 .. joint_7
        # Note: msg.name order might vary
        name_map = dict(zip(msg.name, msg.position))
        for i in range(7):
            joint_name = f"joint_{i+1}"
            if joint_name in name_map:
                self.current_joints[i] = name_map[joint_name]

    def run(self):
        self.logger("Starting Expert Demonstrator...")
        
        # 1. Go to Home (Reset) - Fast
        self.logger("Moving to Start Pose...")
        self.publish_trajectory(self.home_pose, duration=2.0)
        time.sleep(1.0)
        
        # 2. Perform Task: Move to Target - Slow/Smooth
        self.logger("Executing Task: Move to Target...")
        self.publish_trajectory(self.target_pose, duration=5.0)
        
        self.logger("Task Complete.")
        
    def publish_trajectory(self, target, duration):
        start_time = time.time()
        start_joints = self.current_joints.copy()
        
        while time.time() - start_time < duration:
            if rospy.is_shutdown():
                break
                
            elapsed = time.time() - start_time
            alpha = min(elapsed / duration, 1.0)
            
            # Linear interpolation (LERP)
            # In a real expert, we might use a planner or smoothing
            cmd_joints = (1 - alpha) * start_joints + alpha * target
            
            msg = Float32MultiArray()
            msg.data = cmd_joints.tolist()
            self.action_pub.publish(msg)
            
            self.rate.sleep()
            
        # Ensure final pose is sent
        msg = Float32MultiArray()
        msg.data = target.tolist()
        self.action_pub.publish(msg)

if __name__ == '__main__':
    try:
        expert = ExpertDemonstrator()
        # Wait a bit for connections
        time.sleep(2)
        expert.run()
    except rospy.ROSInterruptException:
        pass
