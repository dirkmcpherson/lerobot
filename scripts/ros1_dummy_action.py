#!/usr/bin/env python3
import rospy
import time
import numpy as np
from std_msgs.msg import Float32MultiArray

def publish_dummy_actions():
    rospy.init_node('dummy_action_publisher', anonymous=True)
    pub = rospy.Publisher('/dummy_action', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    
    logger = rospy.loginfo
    logger("Starting dummy action publisher...")
    
    while not rospy.is_shutdown():
        msg = Float32MultiArray()
        # Create a simple sine wave motion for 7 joints
        t = time.time()
        # 7-DOF action
        # Just small movements around 0
        action = np.sin(t) * 0.1
        msg.data = [action] * 7
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_actions()
    except rospy.ROSInterruptException:
        pass
