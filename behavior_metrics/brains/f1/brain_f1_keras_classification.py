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

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'dir1/'

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
        self.gpu_inferencing = True if tf.test.gpu_device_name() else False
        
        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
        else: 
            print("Brain not loaded")

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)
        
    def calculate_v_w(self, prediction):
        '''
            === V ===
            * slow -> 3
            * moderate -> 6
            * fast -> 10
            * very fast -> 13
            === W ===
            * radically left -> 1.7
            * moderate left -> 0.75
            * slightly left -> 0.25
            * slight -> 0 
            * slightly right -> - 0.25
            * moderate right -> - 0.75
            * radically right -> - 1.7
        '''
        
        if prediction == 0:
            # V -> slow W -> radically_left 
            self.motors.sendV(3)
            self.motors.sendW(1.7)
        elif prediction == 1:
            # V -> slow W -> moderately_left
            self.motors.sendV(3)
            self.motors.sendW(0.75)
        elif prediction == 2:
            # V -> slow W -> slightly_left
            self.motors.sendV(3)
            self.motors.sendW(0.25)
        elif prediction == 3:
            # V -> slow W -> slight
            self.motors.sendV(3)
            self.motors.sendW(0)
        elif prediction == 4:
            # V -> slow W -> slightly_right
            self.motors.sendV(3)
            self.motors.sendW(-0.25)
        elif prediction == 5:
            # V -> slow W -> moderately_right
            self.motors.sendV(3)
            self.motors.sendW(-0.75)
        elif prediction == 6:
            # V -> slow W -> radically_right
            self.motors.sendV(3)
            self.motors.sendW(-1.7)
        elif prediction == 7:
            # V -> moderate W -> radically_left
            self.motors.sendV(6)
            self.motors.sendW(1.7)
        elif prediction == 8:
            # V -> moderate W -> moderately_left
            self.motors.sendV(6)
            self.motors.sendW(0.75)
        elif prediction == 9:
            # V -> moderate W -> slightly_left
            self.motors.sendV(6)
            self.motors.sendW(0.25)
        elif prediction == 10:
            # V -> moderate W -> slight
            self.motors.sendV(6)
            self.motors.sendW(0)
        elif prediction == 11:
            # V -> moderate W -> slightly_right
            self.motors.sendV(6)
            self.motors.sendW(-0.25)
        elif prediction == 12:
            # V -> moderate W -> moderately_right
            self.motors.sendV(6)
            self.motors.sendW(-0.75)
        elif prediction == 13:
            # V -> moderate W -> radically_right
            self.motors.sendV(6)
            self.motors.sendW(-1.7)
        elif prediction == 14:
            # V -> fast W -> radically_left
            self.motors.sendV(10)
            self.motors.sendW(1.7)
        elif prediction == 15:
            # V -> fast W -> moderately_left
            # self.motors.sendV(5)
            # self.motors.sendV(7)
            self.motors.sendV(10)
            self.motors.sendW(0.75)
        elif prediction == 16:
            # V -> fast W -> slightly_left
            self.motors.sendV(10)
            self.motors.sendW(0.25)
        elif prediction == 17:
            # V -> fast W -> slight
            self.motors.sendV(10)
            self.motors.sendW(0)
        elif prediction == 18:
            # V -> fast W -> slightly_right
            self.motors.sendV(10)
            self.motors.sendW(-0.25)
        elif prediction == 19:
            # V -> fast W -> moderately_right
            self.motors.sendV(7)
            self.motors.sendW(-0.75)
        elif prediction == 20:
            # V -> fast W -> radically_right
            # self.motors.sendV(5)
            self.motors.sendV(7)
            self.motors.sendW(-1.7)
        elif prediction == 21:
            # V -> very_fast W -> radically_left
            self.motors.sendV(13)
            self.motors.sendW(1.7)
        elif prediction == 22:
            # V -> very_fast W -> moderately_left
            self.motors.sendV(13)
            self.motors.sendW(0.75)
        elif prediction == 23:
            # V -> very_fast W -> slightly_left
            self.motors.sendV(13)
            self.motors.sendW(0.25)
        elif prediction == 24:
            # V -> very_fast W -> slight
            self.motors.sendV(13)
            self.motors.sendW(0)
        elif prediction == 25:
            # V -> very_fast W -> slightly_right
            self.motors.sendV(13)
            self.motors.sendW(-0.25)
        elif prediction == 26:
            # V -> very_fast W -> moderately_right
            self.motors.sendV(13)
            self.motors.sendW(-0.75)
        elif prediction == 27:
            # V -> very_fast W -> radically_right
            self.motors.sendV(13)
            self.motors.sendW(-1.7)

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
         
        self.cont += 1
        
        image = self.camera.getImage().data
        
        if self.cont == 1:
            self.first_image = image

        try:
            image = image[240:480, 0:640]
            img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
            img = np.expand_dims(img, axis=0)
            start_time = time.time()
            prediction = self.net.predict_classes(img)
            self.calculate_v_w(prediction[0])
            self.inference_times.append(time.time() - start_time)
        except Exception as err:
            print(err)
        
        self.update_frame('frame_0', image)

        
