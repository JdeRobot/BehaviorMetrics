#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
import carla
from os import path
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare, GridDropout
)
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from utils.logger import logger
from traceback import print_exc

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'CARLA/'

from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.framework.errors_impl import UnimplementedError
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)


class Brain:

    def __init__(self, sensors, actuators, handler, model, config=None):
        self.camera_0 = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        
        self.cameras_first_images = []

        self.pose = sensors.get_pose3d('pose3d_0')

        self.bird_eye_view = sensors.get_bird_eye_view('bird_eye_view_0')

        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config
        self.inference_times = []
        self.gpu_inference = True if tf.test.gpu_device_name() else False

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()
        
        time.sleep(5)
        self.vehicle = world.get_actors().filter('vehicle.*')[0]

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                logger.info("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            logger.info("** Load TF model **")
            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
            logger.info("** Loaded TF model **")
        else:
            logger.info("** Brain not loaded **")
            logger.info("- Models path: " + PRETRAINED_MODELS)
            logger.info("- Model: " + str(model))

        self.previous_speed = 0
        self.previous_bird_eye_view_image = 0
        self.bird_eye_view_images = 0
        self.bird_eye_view_unique_images = 0


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
        image = self.camera_0.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)
        bird_eye_view_1 = cv2.cvtColor(bird_eye_view_1, cv2.COLOR_BGR2RGB)

        if self.cameras_first_images == []:
            self.cameras_first_images.append(image)
            self.cameras_first_images.append(image_1)
            self.cameras_first_images.append(image_2)
            self.cameras_first_images.append(image_3)
            self.cameras_first_images.append(bird_eye_view_1)

        self.cameras_last_images = [
            image,
            image_1,
            image_2,
            image_3,
            bird_eye_view_1
        ]

        AUGMENTATIONS_TEST = Compose([
            GridDropout(p=1.0)
        ])
        
        bird_eye_view_1 = AUGMENTATIONS_TEST(image=bird_eye_view_1)
        bird_eye_view_1 = bird_eye_view_1["image"]

        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', image_3)

        self.update_frame('frame_0', bird_eye_view_1)
        
        self.update_pose(self.pose.getPose3d())

        image_shape=(66, 200)
        img_base = cv2.resize(bird_eye_view_1, image_shape)

        AUGMENTATIONS_TEST = Compose([
            Normalize()
        ])
        image = AUGMENTATIONS_TEST(image=img_base)
        img = image["image"]

        self.bird_eye_view_images += 1
        if (self.previous_bird_eye_view_image==img).all() == False:
            self.bird_eye_view_unique_images += 1
        self.previous_bird_eye_view_image = img

        velocity_dim = np.full((200, 66), self.previous_speed/30)
        new_img_vel = np.dstack((img, velocity_dim))
        img = new_img_vel

        img = np.expand_dims(img, axis=0)
        start_time = time.time()
        try:
            prediction = self.net.predict(img, verbose=0)
            self.inference_times.append(time.time() - start_time)
            throttle = prediction[0][0]
            steer = prediction[0][1] * (1 - (-1)) + (-1)
            break_command = prediction[0][2]
            speed = self.vehicle.get_velocity()
            vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)
            self.previous_speed = vehicle_speed

            if vehicle_speed < 5:
                self.motors.sendThrottle(1.0)
                self.motors.sendSteer(0.0)
                self.motors.sendBrake(0)
            else:
                self.motors.sendThrottle(throttle)
                self.motors.sendSteer(steer)
                self.motors.sendBrake(break_command)
                
        except NotFoundError as ex:
            logger.info('Error inside brain: NotFoundError!')
            logger.warning(type(ex).__name__)
            print_exc()
            raise Exception(ex)
        except UnimplementedError as ex:
            logger.info('Error inside brain: UnimplementedError!')
            logger.warning(type(ex).__name__)
            print_exc()
            raise Exception(ex)
        except Exception as ex:
            logger.info('Error inside brain: Exception!')
            logger.warning(type(ex).__name__)
            print_exc()
            raise Exception(ex)
            
        
            


