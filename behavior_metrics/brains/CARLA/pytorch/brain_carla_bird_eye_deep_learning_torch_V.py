from PIL import Image
from brains.CARLA.pytorch.utils.pilotnet import PilotNet
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
from torchvision import transforms


import numpy as np

import torch
import torchvision
import cv2
import time
import os
import math
import carla

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'CARLA/'
FLOAT = torch.FloatTensor

class Brain:
    """Specific brain for the CARLA robot. See header."""

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
        self.camera_0 = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        self.bird_eye_view = sensors.get_bird_eye_view('bird_eye_view_0')
        self.handler = handler
        self.cont = 0
        self.inference_times = []
        self.gpu_inference = config['GPU']
        #self.device = torch.device('cuda' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')
        self.first_image = None
        self.transformations = transforms.Compose([
            transforms.Resize((66, 200)),
            transforms.ToTensor(), 
        ])
        
        self.suddenness_distance = []
        self.previous_v = None
        self.previous_w = None

        if config:
            if 'ImageCrop' in config.keys():
                self.cropImage = config['ImageCrop']
            else:
                self.cropImage = True

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            
            if config['UseOptimized']:
                self.net = torch.jit.load(PRETRAINED_MODELS + model).to(self.device)
#                 self.clean_model()
            else:
                self.net = PilotNet((200,66,4), 3).to(self.device)
                self.net.load_state_dict(torch.load(PRETRAINED_MODELS + model,map_location=self.device))
        else: 
            print("Brain not loaded")

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()
        
        time.sleep(5)
        self.vehicle = world.get_actors().filter('vehicle.*')[0]

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if data.shape[0] != data.shape[1]:
            if data.shape[0] > data.shape[1]:
                difference = data.shape[0] - data.shape[1]
                extra_left, extra_right = int(difference/2), int(difference/2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = data.shape[1] - data.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference/2), int(difference/2)
                

            data = np.pad(data, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)), mode='constant', constant_values=0)

        self.handler.update_frame(frame_id, data)
        
    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""
         
        self.cont += 1

        image = self.camera_0.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)
        bird_eye_view_1 = cv2.cvtColor(bird_eye_view_1, cv2.COLOR_BGR2RGB)

        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image)
        self.update_frame('frame_3', image_3)

        self.update_frame('frame_0', bird_eye_view_1)

        try:
            image = Image.fromarray(image)
            image = self.transformations(image)
            image = image / 255.0
            speed = self.vehicle.get_velocity()
            vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

            valor_cuartadimension = torch.full((1, image.shape[1], image.shape[2]), float(vehicle_speed))
            image = torch.cat((image, valor_cuartadimension), dim=0).to(self.device)
            image = image.unsqueeze(0) 
            
            start_time = time.time()
            with torch.no_grad():
                prediction = self.net(image).cpu().numpy() if self.gpu_inference else self.net(image).numpy()
            self.inference_times.append(time.time() - start_time)
            throttle = prediction[0][0]
            steer = prediction[0][1] * (1 - (-1)) + (-1)
            break_command = prediction[0][2]


            if vehicle_speed > 30:
                self.motors.sendThrottle(0)
                self.motors.sendSteer(steer)
                self.motors.sendBrake(break_command)
            else:
                if vehicle_speed < 5:
                    self.motors.sendThrottle(1.0)
                    self.motors.sendSteer(0.0)
                    self.motors.sendBrake(0)
                else:
                    self.motors.sendThrottle(throttle)
                    self.motors.sendSteer(steer)
                    self.motors.sendBrake(break_command)

        except Exception as err:
            print(err)
        
        
