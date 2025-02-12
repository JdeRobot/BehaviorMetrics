#!/usr/bin/env python
# license removed for brevity

import rospy
import numpy as np
import roslib
import tf
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped

from sensor_msgs.msg import Imu

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

import sys
import cv2
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

pub = []

def image_update(msg):
    bridge = CvBridge()
    new_img = Image()
    new_img_cv = bridge.imgmsg_to_cv2(msg,"bgr8")
    new_img = bridge.cv2_to_imgmsg(new_img_cv,"bgr8")
    new_img.header = msg.header
    new_img.header.frame_id = "pylon_camera"
    pub.publish(new_img)

def image_info_update(msg):
    new_img_info = CameraInfo()
    new_img_info = msg
    new_img_info.header.frame_id = "pylon_camera" 
    new_img_info.distortion_model = "plumb_bob"

    # copy the camera info of atlas here
    new_img_info.D = [-0.288902, 0.085422, 0.0013, -6.3e-05, 0.0]
    new_img_info.K = [664.9933754342509, 0.0, 1024.0, 0.0, 664.9933754342509, 768.0, 0.0, 0.0, 1.0]
    new_img_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    new_img_info.P = [664.9933754342509, 0.0, 1024.0, 0.0, 0.0, 664.9933754342509, 768.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    pub_2.publish(new_img_info)

if __name__ == '__main__':
    rospy.init_node('camera_rename')
    # pub = rospy.Publisher('/ada/Lidar32/point_cloud', PointCloud2, queue_size=1)
    pub = rospy.Publisher('/ada/rgb_front/image_color', Image, queue_size=1)
    pub_2 = rospy.Publisher('/ada/rgb_front/camera_info',CameraInfo, queue_size=1)

    #img_sub = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/rgb_front/image_color', Image, image_update)
    img_sub = rospy.Subscriber('/carla/ego_vehicle/rgb_front/image', Image, image_update)
    #img_inf_sub = rospy.Subscriber('/carla/ego_vehicle/camera/rgb/rgb_front/camera_info', CameraInfo, image_info_update)
    img_inf_sub = rospy.Subscriber('/carla/ego_vehicle/rgb_front/camera_info', CameraInfo, image_info_update)


    rospy.spin()
