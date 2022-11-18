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
        self.suddenness_distance = []

        time.sleep(2)


    def execute(self):
        print('EXECUTE')
        #self.motors.sendW(w)
        self.motors.sendV(1)
