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
from brains.f1.torch_utils.pilotnetStacked import PilotNet
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
from collections import deque

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'torch_models/'
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
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu")
        except Exception as e:
            self.device = torch.device("cpu")

        self.gpu_inferencing = torch.cuda.is_available()
        self.first_image = None
        if config:
            if 'ImageCrop' in config.keys():
                self.cropImage = config['ImageCrop']
            else:
                self.cropImage = True

            if 'Horizon' in config.keys():
                self.horizon = config['Horizon']
            else:
                assert False, 'Please specify horizon to use stacked PilotNet brain'

        self.image_horizon = deque([], maxlen=self.horizon)
        self.transformations = transforms.Compose([
                                        transforms.ToTensor()
                                    ])
            
        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = PilotNet((200,66,3), 2, self.horizon).to(self.device)
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
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if self.cont == 1 and image.shape == (480, 640, 3):
            self.first_image = image
        else:
            self.cont = 0

        try:
            if self.cropImage:
                image = image[240:480, 0:640]
            show_image = image
            img = cv2.resize(image, (int(200), int(66)))
            img = Image.fromarray(img)
            start_time = time.time()
            image = self.transformations(img)
            self.image_horizon.append(image)
            if len(self.image_horizon) == self.horizon:
                stacked_image = torch.vstack(list(self.image_horizon))
            else:
                while len(self.image_horizon) < self.horizon:
                    self.image_horizon.append(image)
                stacked_image = torch.vstack(list(self.image_horizon))

            with torch.no_grad():
                stacked_image = FLOAT(stacked_image.unsqueeze(0)).to(self.device)
                prediction = self.net(stacked_image).cpu().numpy()
            self.inference_times.append(time.time() - start_time)
            # prediction_v = prediction[0][0]*6.5
            prediction_v = prediction[0][0]
            prediction_w = prediction[0][1]
            if prediction_w != '' and prediction_w != '':
                self.motors.sendV(prediction_v)
                self.motors.sendW(prediction_w)

        except Exception as err:
            print(err)
        
        self.update_frame('frame_0', show_image)

        
