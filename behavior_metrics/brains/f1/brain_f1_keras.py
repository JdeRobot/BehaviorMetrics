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
    Compose, Normalize
)


PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'tf_models/'

class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
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
        self.gpu_inferencing = True if tf.test.gpu_device_name() else False
        self.config = config
        
        self.deviation_error = []
        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
        else: 
            print("** Brain not loaded **")
            print("- Models path: " + PRETRAINED_MODELS)
            print("- Model: " + str(model))

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)
        
    def check_center(self, position_x):
        if (len(position_x[0]) > 1):
            x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
            not_found = False
        else:
            # The center of the line is in position 326
            x_middle = 326
            not_found = True
        return x_middle, not_found
    
    def get_point(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right)/2 + left
        return int(mid)
    
    def get_deviation_error(self, image_cropped):

        image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_RGB2HSV)
        lower_red = np.array([0,50,50])
        upper_red = np.array([180,255,255])
        image_mask = cv2.inRange(image_hsv, lower_red, upper_red)

        rows, cols = image_mask.shape
        rows = rows - 1     # para evitar desbordamiento

        alt = 0
        ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        if np.count_nonzero(ff[:, 0]) > 0:
            alt = np.min(np.nonzero(ff[:, 0]))

        points = []
        for i in range(3):
            if i == 0:
                index = alt
            else:
                index = rows//(2*i)
            points.append((self.get_point(index, image_mask), index))

        points.append((self.get_point(rows, image_mask), rows))

        position_x_down = np.where(image_mask[points[3][1], :])

        # We see that white pixels have been located and we look if the center is located
        # In this way we can know if the car has left the circuit
        x_middle_left_down, not_found_down = self.check_center(position_x_down)
            
        return abs(326-x_middle_left_down) if not_found_down is False else 327
    

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
         
        self.cont += 1
        
        image = self.camera.getImage().data
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.cont == 1:
            self.first_image = image
            
        self.update_frame('frame_0', image)
        
        try:
            if self.config['ImageCropped']:
                image = image[240:480, 0:640]
            if 'ImageSize' in self.config:
                img = cv2.resize(image, (self.config['ImageSize'][0], self.config['ImageSize'][1]))
            else:
                img = image

            
            deviation_error = self.get_deviation_error(image)
            self.deviation_error.append(deviation_error)
            
            if self.config['ImageNormalized']:
                AUGMENTATIONS_TEST = Compose([
                    Normalize()
                ])
                image = AUGMENTATIONS_TEST(image=img)
                img = image["image"]
                
            
            img = np.expand_dims(img, axis=0)
            start_time = time.time()
            prediction = self.net.predict(img)
            self.inference_times.append(time.time() - start_time)
            
            if self.config['PredictionsNormalized']:
                prediction_v = prediction[0][0]*13
                if prediction[0][1] >= 0.5:
                    x = prediction[0][1] - 0.5
                    prediction_w = x * 6
                else:
                    x = 0.5 - prediction[0][1]
                    prediction_w = x * -6
            else:
                prediction_v = prediction[0][0]
                prediction_w = prediction[0][1]

            if prediction_w != '' and prediction_w != '':
                self.motors.sendV(prediction_v)
                self.motors.sendW(prediction_w)
                
        except Exception as err:
            print(err)
        
