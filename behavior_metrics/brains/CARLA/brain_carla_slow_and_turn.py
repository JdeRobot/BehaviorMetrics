#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)
from utils.constants import DATASETS_DIR, ROOT_PATH



GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR


class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')

        self.pose = sensors.get_pose3d('pose3d_0')

        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.cont = 0
        self.iteration = 0

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None

        time.sleep(2)

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        self.motors.sendThrottle(0.5)
        self.motors.sendSteer(0.05)

        self.update_frame('frame_0', image)
        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', image_3)
        self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())