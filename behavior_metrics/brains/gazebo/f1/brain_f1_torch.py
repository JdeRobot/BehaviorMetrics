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
from brains.gazebo.f1.torch_utils.pilotnet import PilotNet
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path

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
        self.gpu_inference = config['GPU']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')
        self.first_image = None
        self.transformations = transforms.Compose([
                                        transforms.ToTensor()
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
                self.net = PilotNet((200,66,3), 2).to(self.device)
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

    def addPadding(self, img):

        target_height = int(66)
        target_width = int(target_height * img.shape[1]/img.shape[0])
        img_resized = cv2.resize(img, (target_width, target_height))
        padding_left = int((200 - target_width)/2)
        padding_right = 200 - target_width - padding_left
        img = cv2.copyMakeBorder(img_resized.copy(),0,0,padding_left,padding_right,cv2.BORDER_CONSTANT,value=[0, 0, 0])
        return img

    def unnormalize(self, x, min, max):
        return x * (max - min) + min
    
#     def clean_model(self):
#         model = self.net
#         model.eval()
#         remove_attributes = []
#         for key, value in vars(model).items():
#             if value is None:
#                 remove_attributes.append(key)

#         for key in remove_attributes:
#             delattr(model, key)
        
#         self.net = model
        
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
            else:
                image = self.addPadding(image)
            show_image = image
            img = cv2.resize(image, (int(200), int(66)))
            img = Image.fromarray(img)
            image = self.transformations(img).unsqueeze(0)
            image = FLOAT(image).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                prediction = self.net(image).cpu().numpy() if self.gpu_inference else self.net(image).numpy()
            self.inference_times.append(time.time() - start_time)
            
            prediction_v = self.unnormalize(prediction[0][0], min=6.5, max=24)
            prediction_w = self.unnormalize(prediction[0][1], min=-7.1, max=7.1)
            
            if prediction_w != '' and prediction_w != '':
                self.motors.sendV(prediction_v)
                self.motors.sendW(prediction_w)
            
            if self.previous_v != None:
                a = np.array((prediction[0][0], prediction[0][1]))
                b = np.array((self.previous_v, self.previous_w))
                distance = np.linalg.norm(a - b)
                self.suddenness_distance.append(distance)
            self.previous_v = prediction[0][0]
            self.previous_w = prediction[0][1]
            

        except Exception as err:
            print(err)
        
        self.update_frame('frame_0', show_image)

        
