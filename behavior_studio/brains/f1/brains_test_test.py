from behaviorlib.keraslib.keras_predict import KerasPredictor
import cv2
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'dir1/'
MODEL_V = 'test_model_tf_keras.h5'
MODEL_W = 'test_model_tf_keras.h5'

from os import path
import tensorflow as tf
import numpy as np


class Brain:

    def __init__(self, sensors, actuators, handler=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera = sensors.get_camera('camera_0')
        self.handler = handler
        self.cont = 0
        
        if not path.exists(PRETRAINED_MODELS + MODEL_V):
            print("File " + MODEL_V + " cannot be found in " + PRETRAINED_MODELS)
        if not path.exists(PRETRAINED_MODELS + MODEL_W):
            print("File " + MODEL_W + " cannot be found in " + PRETRAINED_MODELS)
            
        self.net_v = tf.keras.models.load_model(PRETRAINED_MODELS + MODEL_V)
        self.net_w = tf.keras.models.load_model(PRETRAINED_MODELS + MODEL_V)
        
    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)
        
    def calculate_w(self, predicted_class):
        """
        Method that calculates the linear speed of the robot (v) based on the predicted class

        The class-speed label conversion for v is as follows:
            class 0 = radically left
            class 1 = moderate left
            class 2 = slightly left
            class 3 = slight
            class 4 = slightly right
            class 5 = moderate right
            class 6 = radically right
        """
        if predicted_class == 0:
            self.motors.sendW(1.7)
        elif predicted_class == 1:
            self.motors.sendW(0.75)
        elif predicted_class == 2:
            self.motors.sendW(0.25)
        elif predicted_class == 3:
            self.motors.sendW(0)
        elif predicted_class == 4:
            self.motors.sendW(-0.25)
        elif predicted_class == 5:
            self.motors.sendW(-0.75)
        elif predicted_class == 6:
            self.motors.sendW(-1.7)
        
    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
        
        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        image = self.camera.getImage().data
        img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        img = np.expand_dims(img, axis=0)
        prediction_v = self.net_v.predict_classes(img)
        prediction_w = self.net_w.predict_classes(img)

        if prediction_w[0] != '' and prediction_w[0] != '':
            self.calculate_w(prediction_v[0])
            self.calculate_w(prediction_w[0])
            #self.motors.sendV(v)
            #self.motors.sendW(w)

        self.update_frame('frame_0', image)

