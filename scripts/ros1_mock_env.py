#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

def mock_env():
    rospy.init_node('mock_env', anonymous=True)
    
    # Publisher for observations
    pub_obs = rospy.Publisher('/mock_obs', Float32MultiArray, queue_size=10)
    # Publisher for actions (simulating a teleoperator or recorded demonstration)
    pub_act = rospy.Publisher('/mock_action', Float32MultiArray, queue_size=10)
    # Publisher for mock images
    from sensor_msgs.msg import Image
    pub_img = rospy.Publisher('/mock_image', Image, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz => 1 step per 0.1s
    
    # Trajectory parameters
    start_pos = np.array([-1.0, -1.0, -1.0])
    end_pos = np.array([1.0, 1.0, 0.0])
    steps = 10
    
    step_count = 0
    
    rospy.loginfo("Starting mock environment with consistent trajectory...")
    
    while not rospy.is_shutdown():
        # Calculate interpolation factor alpha [0, 1]
        # We want to go from start to end in 10 steps.
        # Let's say loop is 20 steps total, 10 moving, 10 reset/wait?
        # Or just loop the 10 steps continuously?
        # "moving ... over ten steps"
        
        # Current index in the trajectory (0 to steps)
        idx = step_count % (steps + 5) # Add 5 steps pause at end
        
        if idx <= steps:
            alpha = idx / float(steps)
            current_pos = (1 - alpha) * start_pos + alpha * end_pos
            
            # Action: The target position for the *next* step.
            # Ideally A[t] leads to S[t+1].
            # If we are at S[t], action should be S[t+1].
            if idx < steps:
                next_alpha = (idx + 1) / float(steps)
                next_pos = (1 - next_alpha) * start_pos + next_alpha * end_pos
            else:
                next_pos = end_pos # Stay at end
                
            # Add small noise to observation to make it realistic
            # User asked for "structured noisy data" but "consistent policy"
            obs_noise = np.random.normal(0, 0.01, size=3)
            # action noise? maybe less noise on action
            
            obs_data = current_pos + obs_noise
            act_data = next_pos 
            
            msg_obs = Float32MultiArray()
            msg_obs.data = obs_data.tolist()
            
            msg_act = Float32MultiArray()
            msg_act.data = act_data.tolist()

            # Create mock image (black with moving white square)
            img_h, img_w = 96, 96
            img_data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # Draw a square based on position (mapped to pixel coords)
            # Map [-1, 1] to [0, 96]
            u = int((current_pos[0] + 1) / 2 * (img_w - 10))
            v = int((current_pos[1] + 1) / 2 * (img_h - 10))
            u = np.clip(u, 0, img_w - 10)
            v = np.clip(v, 0, img_h - 10)
            img_data[v:v+10, u:u+10, :] = 255
            
            msg_img = Image()
            msg_img.header.stamp = rospy.Time.now()
            msg_img.height = img_h
            msg_img.width = img_w
            msg_img.encoding = "rgb8"
            msg_img.is_bigendian = 0
            msg_img.step = img_w * 3
            msg_img.data = img_data.tobytes()
            
            pub_obs.publish(msg_obs)
            pub_act.publish(msg_act)
            pub_img.publish(msg_img)
            
            rospy.loginfo_throttle(1, f"Step {idx}: Obs={obs_data}, Act={act_data}")
        else:
            # Pause / Reset phase
            if idx == (steps + 5) - 1:
               rospy.loginfo("Resetting trajectory...") 
        
        step_count += 1
        rate.sleep()

if __name__ == '__main__':
    try:
        mock_env()
    except rospy.ROSInterruptException:
        pass
