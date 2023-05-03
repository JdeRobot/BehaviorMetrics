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

    def get_center(self, lines):
        #print('ENTRA!')
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            print(f"No lines detected in the image")
            return 0

    def execute(self):
        image_2 = self.camera_2.getImage().data
        hsv_nemo = cv2.cvtColor(image_2, cv2.COLOR_RGB2HSV)


        unique_values=set( tuple(v) for m2d in hsv_nemo for v in m2d )
        #print(unique_values)
        #print(len(unique_values))



        light_sidewalk = (151, 217, 243)
        dark_sidewalk = (153, 219, 245)

        light_pavement = (149, 127, 127)
        dark_pavement = (151, 129, 129)

        mask_sidewalk = cv2.inRange(hsv_nemo, light_sidewalk, dark_sidewalk)
        result_sidewalk = cv2.bitwise_and(image_2, image_2, mask=mask_sidewalk)

        mask_pavement = cv2.inRange(hsv_nemo, light_pavement, dark_pavement)
        result_pavement = cv2.bitwise_and(image_2, image_2, mask=mask_pavement)


        # Adjust according to your adjacency requirement.
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilating masks to expand boundary.
        color1_mask = cv2.dilate(mask_sidewalk, kernel, iterations=1)
        color2_mask = cv2.dilate(mask_pavement, kernel, iterations=1)

        # Required points now will have both color's mask val as 255.
        common = cv2.bitwise_and(color1_mask, color2_mask)
        SOME_THRESHOLD = 10

        # Common is binary np.uint8 image, min = 0, max = 255.
        # SOME_THRESHOLD can be anything within the above range. (not needed though)
        # Extract/Use it in whatever way you want it.
        intersection_points = np.where(common > SOME_THRESHOLD)

        # Say you want these points in a list form, then you can do this.
        pts_list = [[r, c] for r, c in zip(*intersection_points)]
        #print(pts_list)

        #for x, y in pts_list:
        #    image_2[x][y] = (255, 0, 0)

        red_line_mask = np.zeros((70, 400, 3), dtype=np.uint8)

        for x, y in pts_list:
            red_line_mask[x][y] = (255, 0, 0)


        ##########################################################################################

        #x_row = [10,60,110] 
        x_row = [10,30,60]


        x_row = [50, 60, 69]

        mask = red_line_mask

        print(mask.shape)

        lines = [mask[x_row[idx], :] for idx, x in enumerate(x_row)]
        #print(lines)
        #print(lines[0])
        #print(type(lines[0]))
        #print(lines[0].shape)
        unique_values=set( tuple(m2d) for m2d in lines[0] )
        unique_values=set( tuple(m2d) for m2d in lines[1] )
        unique_values=set( tuple(m2d) for m2d in lines[2] )

        #centrals = list(map(self.get_center, lines))

        #point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
        #print(point)
        #a = np.min(np.nonzero(lines)) + point
        #print(a)

        centrals = list(map(self.get_center, lines))
        print(centrals)

        for idx, central in enumerate(centrals):
            red_line_mask[idx][centrals[0]] = (0, 255, 0)


        ##########################################################################################

        self.update_frame('frame_0', red_line_mask)
        self.update_frame('frame_1', result_pavement)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', result_sidewalk)
        

