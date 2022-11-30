#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
import carla
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)
from utils.constants import DATASETS_DIR, ROOT_PATH

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

PRETRAINED_MODELS = "../models/"

class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')

        self.pose = sensors.get_pose3d('pose3d_0')

        self.bird_eye_view = sensors.get_bird_eye_view('bird_eye_view_0')

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

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()
        time.sleep(3)
        self.vehicle = world.get_actors().filter('vehicle.*')[0]

        model = '/home/jderobot/Documents/Projects/BehaviorMetrics/PlayingWithCARLA/models/20221104-143528_pilotnet_CARLA_17_10_dataset_bird_eye_300_epochs_no_flip_3_output_velocity_all_towns_vel_30_cp.h5'
        self.net = tf.keras.models.load_model(model)
        self.previous_speed = 0


    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if data.shape[0] != data.shape[1]:
            if data.shape[0] > data.shape[1]:
                difference = data.shape[0] - data.shape[1]
                extra_left, extra_right = int(difference/2), int(difference/2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = data.shape[1] - data.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference/2), int(difference/2)
                

            data = np.pad(data, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)), mode='constant', constant_values=0)
            
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        image = self.camera.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)

        #print(self.bird_eye_view.getImage(self.vehicle))

        #self.update_frame('frame_0', image)
        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', image_3)

        self.update_frame('frame_0', bird_eye_view_1)
        
        self.update_pose(self.pose.getPose3d())

        image_shape=(50, 150)
        img_base = cv2.resize(bird_eye_view_1, image_shape)

        AUGMENTATIONS_TEST = Compose([
            Normalize()
        ])
        image = AUGMENTATIONS_TEST(image=img_base)
        img = image["image"]

        #velocity_dim = np.full((150, 50), 0.5)
        velocity_dim = np.full((150, 50), self.previous_speed/30)
        new_img_vel = np.dstack((img, velocity_dim))
        img = new_img_vel

        img = np.expand_dims(img, axis=0)
        prediction = self.net.predict(img, verbose=0)
        throttle = prediction[0][0]
        steer = prediction[0][1] * (1 - (-1)) + (-1)
        break_command = prediction[0][2]
        speed = self.vehicle.get_velocity()
        vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
        self.previous_speed = vehicle_speed

        if vehicle_speed > 300:
            self.motors.sendThrottle(0)
            self.motors.sendSteer(steer)
            self.motors.sendBrake(0)
        else:
            if vehicle_speed < 2:
                self.motors.sendThrottle(1.0)
                self.motors.sendSteer(0.0)
                self.motors.sendBrake(0)
            else:
                self.motors.sendThrottle(throttle)
                self.motors.sendSteer(steer)
                self.motors.sendBrake(0)
            


