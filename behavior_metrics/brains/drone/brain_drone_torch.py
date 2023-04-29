"""
    Robot: drone
    Framework: torch
    Number of networks: 1
    Network type: None
    Predicionts:
        linear speed(v)
        angular speed(w)
        z-velocity (vz)

"""

import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import time
import os
from PIL import Image
from brains.drone.torch_utils.deeppilot import DeepPilot
from drone_wrapper import DroneWrapper
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
from collections import deque

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'torch_drone_models/'
FLOAT = torch.FloatTensor

class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, model=None, handler=None, config=None):
        """Constructor of the class.

        Arguments:
            sensors {robot.sensors.Sensors} -- Sensors instance of the robot
            actuators {robot.actuators.Actuators} -- Actuators instance of the robot

        Keyword Arguments:
            handler {brains.brain_handler.Brains} -- Handler of the current brain. Communication with the controller
            (default: {None})
        """
        self.drone = DroneWrapper()
        self.handler = handler
        # self.drone.takeoff()
        self.takeoff = False
        self.speed_history = deque([], maxlen=100)
        self.speedz_history = deque([0]*100, maxlen=100)
        self.rot_history = deque([], maxlen=1)

        self.handler = handler
        self.cont = 0
        self.iteration = 0
        self.inference_times = []
        self.device = torch.device("cpu")
        self.gpu_inferencing = torch.cuda.is_available()
        self.first_image = None
        self.transformations = transforms.Compose([
                                        transforms.ToTensor()
                                    ])

        if config:
            if 'ImageCrop' in config.keys():
                self.cropImage = config['ImageCrop']
            else:
                self.cropImage = True

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = DeepPilot((224,224,3), 3).to(self.device)
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

    def getPose3d(self):
        return self.drone.get_position()

    def addPadding(self, img):

        target_height = int(224)
        target_width = int(target_height * img.shape[1]/img.shape[0])
        img_resized = cv2.resize(img, (target_width, target_height))
        padding_left = int((224 - target_width)/2)
        padding_right = 224 - target_width - padding_left
        img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
        return img


    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
         
        self.cont += 1
        
        if self.iteration == 0 and not self.takeoff:
            self.drone.takeoff()
            self.takeoff = True
            self.initial_flight_done = False

        img_frontal = self.drone.get_frontal_image()
        img_ventral = self.drone.get_ventral_image()

        if self.cont == 1 and img_frontal.shape == (3, 3, 3) or img_ventral.shape == (3, 3, 3):
            time.sleep(3)
            self.cont = 0
        else:
            self.first_image = img_frontal

        self.update_frame('frame_0', img_frontal)
        self.update_frame('frame_1', img_ventral)
        
        image = img_frontal
        
        try:
            if self.cropImage:
                image = image[120:240,0:320]
            else:
                image = self.addPadding(image)
            show_image = image
            X = image.copy()    
            if X is not None:
                X = cv2.resize(X, (224, 224))
                X = np.transpose(X,(2,0,1))
                X = np.squeeze(X)
                X = np.transpose(X, (1,2,0))
            img = Image.fromarray(X)
            start_time = time.time()
            with torch.no_grad():
                image = self.transformations(img).unsqueeze(0)
                image = FLOAT(image).to(self.device)
                prediction = self.net(image).numpy()
            self.inference_times.append(time.time() - start_time)
        
            prediction_v = prediction[0][0]
            prediction_w = prediction[0][1]
            prediction_vz = prediction[0][2]
            if prediction_w != '' and prediction_w != '' and prediction_vz != '':
                self.speed_history.append(prediction_v)
                self.speedz_history.append(prediction_w)
                self.rot_history.append(prediction_vz)

                speed_cmd = np.mean(self.speed_history)
                speed_z_cmd = np.clip(np.mean(self.speedz_history),-2,2)
                rotation_cmd = np.mean(self.rot_history)

                self.drone.set_cmd_vel(speed_cmd, 0, speed_z_cmd, rotation_cmd)

                self.iteration += 1


        except Exception as err:
            print(err)
        
        self.update_frame('frame_0', show_image)

        
