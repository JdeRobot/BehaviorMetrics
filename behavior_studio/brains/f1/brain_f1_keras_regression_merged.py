"""
    Robot: F1
    Framework: keras
    Number of networks: 2
    Network type: Regression
    Predicionts:
        linear speed(v)
        angular speed(w)

    This brain uses regression networks based on Keras framework to predict the linear and angular velocity
    of the F1 car. For that task it uses two different regression convolutional neural networks, one for v
    and another one for w
"""

import tensorflow as tf
import numpy as np
import cv2
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
import time

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'dir1/'


# MODEL_PILOTNET = 'model_pilotnet_cropped_300.h5' # CHANGE TO YOUR NET
# MODEL_PILOTNET = 'merged_model_pilotnet_cropped_100_dense_1.h5'
# MODEL_PILOTNET = 'merged_model_pilotnet_cropped_100_dense_2.h5'
# MODEL_PILOTNET = 'merged_model_tinypilotnet_cropped_100.h5'
MODEL_PILOTNET = 'merged_model_tinypilotnet_cropped_300.h5'


from os import path

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
        
        if not path.exists(PRETRAINED_MODELS + MODEL_PILOTNET):
            print("File "+MODEL_PILOTNET + " cannot be found in " + PRETRAINED_MODELS)
            
        self.net = tf.keras.models.load_model(PRETRAINED_MODELS + MODEL_PILOTNET)

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
        try:
            image = image[240:480, 0:640]
            img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
            img = np.expand_dims(img, axis=0)

            start_time = time.time()
            prediction = self.net.predict(img)
            self.inference_times.append(time.time() - start_time)
            # prediction_v = prediction[0][0] * 0.5
            # prediction_v = prediction[0][0] * 0.4
            prediction_v = prediction[0][0] * 0.5
            prediction_w = prediction[0][1]

            if prediction_w != '' and prediction_w != '':
                self.motors.sendV(prediction_v)
                self.motors.sendW(prediction_w)
        except:
            print('---ERROR---')
            pass

        self.update_frame('frame_0', image)
