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
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config

        time.sleep(2)

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """

        self.handler.update_frame(frame_id, data)


    def execute(self):
        image = self.camera.getImage().data

        v, w = 0, 0
        self.motors.sendV(v)
        self.motors.sendW(w)

        self.update_frame('frame_0', image)
        
