import time
from typing import Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from sensor_msgs.msg import Image

from brains.gazebo.f1.rl_utils.settings import QLearnConfig
from brains.gazebo.f1.rl_utils.image_f1 import ListenerCamera
from brains.gazebo.f1.rl_utils.models.f1_env import F1Env



class QlearnF1FollowLineEnvGazebo(F1Env):
    def __init__(self, **config):
        F1Env.__init__(self, **config)
        self.image = ListenerCamera("/F1ROS/cameraL/image_raw")
        self.previous_image = self.image.getImage().data
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(3)
        self.config = QLearnConfig()

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    @staticmethod
    def get_center(lines):
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            print(f"No lines detected in the image")
            return 0

    def processed_image(self, img: Image) -> list:
        """
        - Convert img to HSV.
        - Get the image processed.
        - Get 3 lines from the image.
        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        img_sliced = img[240:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(
            img_proc, (0, 30, 30), (0, 255, 255)
        )  # default: 0, 30, 30 - 0, 255, 200
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [
            mask[self.config.x_row[idx], :] for idx, x in enumerate(self.config.x_row)
        ]
        centrals = list(map(self.get_center, lines))

        return centrals

    def calculate_observation(self, state: list) -> list:
        normalize = 40
        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((self.config.center_image - x) / normalize) + 1)

        return final_state

    def step(self, action) -> Tuple:
        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        start = time.time()
        f1_image_camera = self.image.getImage()

        while np.array_equal(self.previous_image, f1_image_camera.data):
            if (time.time() - start) > 0.1:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
            f1_image_camera = self.image.getImage()
        
        self.previous_image = f1_image_camera.data

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)

        done = False
        reward = 0

        return state, reward, done, {}

    def reset(self):
        self._gazebo_reset()


        # Get camera info
        start = time.time()
        f1_image_camera = self.image.getImage()

        while np.array_equal(self.previous_image, f1_image_camera.data):
            if (time.time() - start) > 0.1:
                vel_cmd = Twist()
                vel_cmd.linear.x = 0
                vel_cmd.angular.z = 0
                self.vel_pub.publish(vel_cmd)
            f1_image_camera = self.image.getImage()
        
        self.previous_image = f1_image_camera.data

        points = self.processed_image(f1_image_camera.data)
        state = self.calculate_observation(points)

        return state
