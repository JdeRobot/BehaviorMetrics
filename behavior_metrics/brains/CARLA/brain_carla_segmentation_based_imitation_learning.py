from torchvision import transforms
from PIL import Image
from brains.CARLA.utils.pilotnet_onehot import PilotNetOneHot
from brains.CARLA.utils.test_utils import traffic_light_to_int, model_control
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from brains.CARLA.utils.high_level_command import HighLevelCommandLoader
from os import path

import numpy as np

import torch
import torchvision
import cv2
import time
import os
import math
import carla

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'il_models/'

class Brain:

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera_rgb = sensors.get_camera('camera_0') # rgb front view camera
        self.camera_seg = sensors.get_camera('camera_2') # segmentation camera
        self.handler = handler
        self.inference_times = []
        self.gpu_inference = config['GPU']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(100.0)
        world = client.get_world()
        self.map = world.get_map()

        weather = carla.WeatherParameters.ClearNoon
        world.set_weather(weather)

        self.vehicle = None
        while self.vehicle is None:
            for vehicle in world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == 'ego_vehicle':
                    self.vehicle = vehicle
                    break
            if self.vehicle is None:
                print("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again
        
        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            
            if config['UseOptimized']:
                self.net = torch.jit.load(PRETRAINED_MODELS + model).to(self.device)
            else:
                self.net = PilotNetOneHot((288, 200, 6), 3, 4, 4).to(self.device)
                self.net.load_state_dict(torch.load(PRETRAINED_MODELS + model,map_location=self.device))
                self.net.eval()
        
        self.hlc_loader = HighLevelCommandLoader(self.vehicle, self.map)
            
    
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
         
        rgb_image = self.camera_rgb.getImage().data
        seg_image = self.camera_seg.getImage().data
        
        self.update_frame('frame_0', rgb_image)
        self.update_frame('frame_1', seg_image)
        
        start_time = time.time()
        try:
            # calculate speed
            speed_m_s = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(speed_m_s.x**2 + speed_m_s.y**2 + speed_m_s.z**2)
            
            # randomly choose high-level command if at junction
            hlc = self.hlc_loader.get_random_hlc()

            # get traffic light status
            light_status = -1
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                light_status = traffic_light.get_state()

            print(f'hlc: {hlc}')
            #print(f'light: {light_status}')
            frame_data = {
                'hlc': hlc,
                'measurements': speed,
                'rgb': np.copy(rgb_image),
                'segmentation': np.copy(seg_image),
                'light': np.array([traffic_light_to_int(light_status)])
            }

            throttle, steer, brake = model_control(self.net, 
                                    frame_data, 
                                    ignore_traffic_light=False, 
                                    device=self.device, 
                                    combined_control=False)
            
            self.inference_times.append(time.time() - start_time)

            self.motors.sendThrottle(throttle)
            self.motors.sendSteer(steer)
            self.motors.sendBrake(brake)
        except Exception as err:
            print(err)