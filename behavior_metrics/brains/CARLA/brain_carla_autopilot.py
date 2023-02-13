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



GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR


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
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()
        time.sleep(3)
        self.vehicle = world.get_actors().filter('vehicle.*')[0]

        self.counter = 0

        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        traffic_speed_limits = world.get_actors().filter('traffic.speed_limit*')
        print(traffic_speed_limits)
        for traffic_light in traffic_lights:
            traffic_light.set_green_time(20000)
            traffic_light.set_state(carla.TrafficLightState.Green)

        for speed_limit in traffic_speed_limits:
            success = speed_limit.destroy()
            print(success)


        route = ["Straight", "Straight", "Straight", "Straight", "Straight",
        "Straight", "Straight", "Straight", "Straight", "Straight",
        "Straight", "Straight", "Straight", "Straight", "Straight",
        "Straight", "Straight", "Straight", "Straight", "Straight",
        "Straight", "Straight", "Straight", "Straight", "Straight",
        "Straight", "Straight", "Straight", "Straight", "Straight"]
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_route(self.vehicle, route)
        self.vehicle.set_autopilot(True)
        

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

        bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)
        bird_eye_view_1 = cv2.cvtColor(bird_eye_view_1, cv2.COLOR_BGR2RGB)

        #print(self.bird_eye_view.getImage(self.vehicle))

        self.update_frame('frame_0', image)
        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', image_3)

        #self.update_frame('frame_0', bird_eye_view_1)
        

        self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())

