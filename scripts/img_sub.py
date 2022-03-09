#!/usr/bin/env python3

import cv2
import numpy as np

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def Detector(img_data):
    bridge = CvBridge()
    try:
        cv_img = bridge.imgmsg_to_cv2(img_data, desired_encoding='bgr8')
    except CvBridgeError as e:
        print(e)
    
    # By numpy method
    #cv_img = np.frombuffer(img_data.data, dtype=np.uint8).reshape(img_data.height, img_data.width, -1)
    
    cv2.imshow("D435i Image", cv_img)
    cv2.waitKey(3)
    repub_image.publish(img_data)

if __name__ == '__main__':
    rospy.init_node('DoorDetector')
    
    rospy.Subscriber('/camera/color/image_raw', Image, Detector)
    repub_image = rospy.Publisher('/re_image', Image, queue_size=1)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
    