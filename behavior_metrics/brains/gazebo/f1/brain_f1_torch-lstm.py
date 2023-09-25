"""
    Robot: F1
    Framework: torch
    Number of networks: 1
    Network type: None
    Predicionts:
        linear speed(v)
        angular speed(w)

"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import time
import os
from PIL import Image
from brains.gazebo.f1.torch_utils.deepest_lstm_tinypilotnet import DeepestLSTMTinyPilotNet
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
from albumentations import (
    Compose, Normalize
)


PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'gazebo/torch_models/'
FLOAT = torch.FloatTensor

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
        self.device = torch.device("cpu")
        self.gpu_inference = torch.cuda.is_available()
        self.transformations = transforms.Compose([
                                        transforms.ToTensor()
                                    ])
        self.config = config
    
        if config:
            if 'ImageCrop' in config.keys():
                self.cropImage = config['ImageCrop']
            else:
                self.cropImage = True

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = DeepestLSTMTinyPilotNet((50,100,3), 2).to(self.device)
            self.net.load_state_dict(torch.load(PRETRAINED_MODELS + model,map_location=self.device))
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
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.cont == 1:
            self.first_image = image

        self.update_frame('frame_0', image)

        try:
            if self.cropImage:
                image = image[240:480, 0:640]
            if 'ImageSize' in self.config:
                img = cv2.resize(image, (self.config['ImageSize'][0], self.config['ImageSize'][1]))
            else:
                img = image

            img = Image.fromarray(img)
            start_time = time.time()
            with torch.no_grad():
                image = self.transformations(img).unsqueeze(0)
                image = FLOAT(image).to(self.device)
                prediction = self.net(image).numpy()
            self.inference_times.append(time.time() - start_time)

            if self.config['PredictionsNormalized']:
                prediction_v = prediction[0][0] * 6
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
