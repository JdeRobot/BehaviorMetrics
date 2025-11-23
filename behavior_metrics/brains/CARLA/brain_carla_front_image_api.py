#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import threading
import time
from utils.constants import DATASETS_DIR, ROOT_PATH
import cv2 as cv
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from brains.CARLA.utils.pilotnet import PilotNet
import timm
GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR


class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        
        # Obtain sensors
        self.camera = sensors.get_camera('camera_0')
        self.pose = sensors.get_pose3d('pose3d_0')
        # Get actuators
        self.motors = actuators.get_motor('motors_0')
        
        self.handler = handler
        self.config = config

        self.cont = 0
        self.iteration = 0
        
        # Create model object
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess = transforms.Compose([
            transforms.ToTensor()
        ])

        self.input_size = config['ImageSize']


        if config['ModelName'] == 'mobilenet_large':
            self.model = models.mobilenet_v3_large()
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, 2)
        elif config['ModelName'] == 'mobilenet_small':
            self.model = models.mobilenet_v3_small()
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, 2)
        elif config['ModelName'] == 'resnet':
            self.model = models.resnet18()
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
        elif config['ModelName'] == 'efficientnet_v2':
            self.model = models.efficientnet_v2_s(weights=None)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = torch.nn.Linear(num_ftrs, 2)
        elif config['ModelName'] == 'efficientvit':
            model = timm.create_model('efficientvit_b0', pretrained=False)
            num_ftrs = model.head.classifier[-1].in_features
            model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
        elif config['ModelName'] == 'fastvit':
            model = timm.create_model('fastvit_mci0', pretrained=False)
            num_ftrs = model.head.classifier[-1].in_features
            model.head.classifier[-1] = nn.Linear(num_ftrs, 2)
        else:
            self.model = PilotNet(self.input_size, 2)
        # Load the state dictionary from the local .pth file
        state_dict = torch.load(config['SavedModelPath'], weights_only=True)
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)

        # Move the model to the selected device (cpu or gpu)
        self.model.to(self.device)
    
        # Set the model to evaluation mode
        self.model.eval()

        time.sleep(2)

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        image = self.camera.getImage()
        if image is not None:
            #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            cropped_image = image[240:480, 0:640]
            resized_image = cv.resize(cropped_image, (int(self.input_size[1]), int(self.input_size[0])))
            input_tensor = self.preprocess(resized_image).to(self.device)
            input_batch = input_tensor.unsqueeze(0)
            output = self.model(input_batch)
            if self.device == "cpu":
                net_throttle = output[0].detach().numpy()[0].item()
                net_steer = output[0].detach().numpy()[1].item()
            else:
                net_throttle = output.data.cpu().numpy()[0][0].item()
                net_steer = output.data.cpu().numpy()[0][1].item()
            
            self.motors.sendThrottle(net_throttle)
            print(net_steer)
            self.motors.sendSteer(net_steer)
            self.update_frame('frame_0', image)
        self.update_pose(self.pose.getPose3d())
        #print(self.pose.getPose3d())
