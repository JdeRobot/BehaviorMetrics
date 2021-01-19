"""
    Robot: F1
    Framework: keras
    Number of networks: 2
    Network type: LSTM
    Predicionts:
        linear speed(v)
        angular speed(w)

    This brain uses LSTM networks based on Keras framework to predict the linear and angular velocity
    of the F1 car. For that task it uses two different LSTM convolutional neural networks, one for v
    and another one for w
"""

import tensorflow as tf
import numpy as np
import cv2
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
import time
from os import path
import os

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'behavior-studio-volume/'

MODEL_LSTM_V = 'model_lstm_tinypilotnet_cropped_150_v.h5' # CHANGE TO YOUR NET
MODEL_LSTM_W = 'model_lstm_tinypilotnet_cropped_150_w.h5' # CHANGE TO YOUR NET


class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, sensors, actuators, handler=None):
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
        #os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if tf.test.gpu_device_name():
            print('------------------------------------- GPU found ------------------------------------- ')
        else:
            print("------------------------------------- No GPU found ------------------------------------- ")    
        self.gpu_inferencing = True if tf.test.gpu_device_name() else False
        
        if not path.exists(PRETRAINED_MODELS + MODEL_LSTM_V):
            print("File "+MODEL_LSTM_V+" cannot be found in " + PRETRAINED_MODELS)
        if not path.exists(PRETRAINED_MODELS + MODEL_LSTM_W):
            print("File "+MODEL_LSTM_W+" cannot be found in " + PRETRAINED_MODELS)
            
        self.net_v = tf.keras.models.load_model(PRETRAINED_MODELS + MODEL_LSTM_V)
        self.net_w = tf.keras.models.load_model(PRETRAINED_MODELS + MODEL_LSTM_W)

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
        # Normal image size -> (160, 120)
        # Cropped image size -> (60, 160)
        
        # NORMAL IMAGE
        #print((int(image.shape[1] / 4), int(image.shape[0] / 4)))
        #img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        
        # CROPPED IMAGE
        image = image[240:480, 0:640]
        img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        img = np.expand_dims(img, axis=0)
        start_time = time.time()
        prediction_v = self.net_v.predict(img)
        prediction_v = prediction_v * 0.5
        prediction_w = self.net_w.predict(img)
        self.inference_times.append(time.time() - start_time)
        
        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(prediction_v)
            self.motors.sendW(prediction_w)

        self.update_frame('frame_0', image)
