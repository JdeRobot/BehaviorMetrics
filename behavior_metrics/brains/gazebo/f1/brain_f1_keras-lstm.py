"""
    Robot: F1
    Framework: keras
    Number of networks: 1
    Network type: None
    Predicionts:
        linear speed(v)
        angular speed(w)

"""

import tensorflow as tf
import numpy as np
import cv2
import time
import os

from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, 
    HueSaturationValue, FancyPCA, RandomGamma, 
    GaussianBlur, ToFloat, Normalize, PadIfNeeded
)

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'behavior-studio-volume/'

class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, sensors, actuators, model=None, handler=None):
        """Constructor of the class.

        Arguments:
            sensors {robot.sensors.Sensors} -- Sensors instance of the robot
            actuators {robot.actuators.Actuators} -- Actuators instance of the robot

        Keyword Arguments:
            handler {brains.brain_handler.Brains} -- Handler of the current brain. Communication with the controller
            (default: {None})
        """
        self.motors = actuators.get_motor('motors_0')
        self.camera = sensors.get_camera('camera_0')
        self.handler = handler
        self.cont = 0
        self.inference_times = []
        self.gpu_inference = True if tf.test.gpu_device_name() else False
        self.previous_images = np.zeros((1, 50, 100, 3))
        print(self.previous_images.shape)
        print(self.previous_images.shape)
        print(self.previous_images.shape)
        print(self.previous_images.shape)
        print(self.previous_images.shape)
        print('---- MODEL---')
        model = '20210727-161424_deepest_lstm_tinypilotnet_pilotnet_model_100_all_crop_cp.h5'
        print(model)
        print('-.-.-.-.-.-.-.-.-')
        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
            self.net.summary()
            lkasbdflasbflkdasbflkjsaf
            exit()
        else: 
            print("Brain not loaded")

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
         
        self.cont += 1
        
        image = self.camera.getImage().data
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.cont == 1:
            self.first_image = image

        try:
            # ValueError: Input 0 is incompatible with layer time_distributed_9: expected shape=(None, None, 100, 50, 3), found shape=(None, 3, 50, 100, 3)
            # ValueError: Input 0 is incompatible with layer time_distributed_9: expected shape=(None, None, 100, 50, 3), found shape=(None, 1, 50, 100, 3)
            image = image[240:480, 0:640]
            #img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
            #img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 2)))
            
            #image_shape=(200, 66)
            #image_shape=(66, 200)
            #image_shape = (66, 66)
            image_shape=(100, 50)
            img = cv2.resize(image, image_shape)
            
            # img_shape = (60, 160, 3)
            # img_shape = (120, 160, 3)
            
            #img = image
            #print(img.shape)
            #print(img)
            #img = np.expand_dims(img, axis=0)
            #print(img.shape)
            AUGMENTATIONS_TEST = Compose([
                Normalize()
            ])
            
            print(1)
            print(self.previous_images.shape)
            self.previous_images = self.previous_images[1:1:]
            img = np.expand_dims(img, axis=0)
            image = AUGMENTATIONS_TEST(image=img)
            img = image["image"]
            print(2)
            print(self.previous_images.shape)
            print(img.shape)
            self.previous_images = np.append(self.previous_images, img, axis=0)
            print(3)
            #print(self.previous_images.shape)
            print(4)
            #print(len(self.previous_images))
            #print(self.previous_images[2])
            print(5)

            img_points = np.expand_dims(self.previous_images[0], axis=0)
            img_points = np.expand_dims(img_points, axis=0)
            #print(img_points[0])
            print(img_points.shape)
            print(self.previous_images.shape)
            print(6)

            start_time = time.time()
            prediction = self.net.predict(img_points)
            # prediction = self.net.predict(self.previous_images)
            print(prediction)
            self.inference_times.append(time.time() - start_time)
            #prediction_v = prediction[0][0]*3
            prediction_v = prediction[0][0]*6
            if prediction[0][1] >= 0.5:
                x = prediction[0][1] - 0.5
                prediction_w = x * 6
            else:
                x = 0.5 - prediction[0][1]
                prediction_w = x * -6
            # prediction_w = prediction[0][1]*3
            print('v -> ' + str(prediction_v) + ' w -> ' + str(prediction_w))
            if prediction_v != '' and prediction_w != '':
                self.motors.sendV(prediction_v)
                self.motors.sendW(prediction_w)

        except Exception as err:
            print(err)
        
        self.update_frame('frame_0', image)

        
