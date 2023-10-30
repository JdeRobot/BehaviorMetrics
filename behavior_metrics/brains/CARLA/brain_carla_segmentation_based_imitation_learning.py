from brains.CARLA.utils.pilotnet_onehot import PilotNetOneHot
from brains.CARLA.utils.test_utils import traffic_light_to_int, model_control, calculate_delta_yaw
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from brains.CARLA.utils.high_level_command import HighLevelCommandLoader
from os import path

import numpy as np

import torch
import time
import math
import carla

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'CARLA/'

class Brain:

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera_rgb = sensors.get_camera('camera_0') # rgb front view camera
        self.camera_seg = sensors.get_camera('camera_2') # segmentation camera
        self.handler = handler
        self.inference_times = []
        self.gpu_inference = config['GPU']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')
        self.red_light_counter = 0
        self.running_light = False

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
        
        if 'Route' in config:
            route = config['Route']
            print('route: ', route)
        else: 
            route = None
            
        self.hlc_loader = HighLevelCommandLoader(self.vehicle, self.map, route=route)
        self.prev_hlc = 0
        self.prev_yaw = None
        self.delta_yaw = 0

        self.target_point = None
        self.termination_code = 0 # 0: not terminated; 1: arrived at target; 2: wrong turn
    
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
            velocity = self.vehicle.get_velocity()
            speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed = 3.6 * speed_m_s #m/s to km/h 
            
            # read next high-level command or choose a random direction
            hlc = self.hlc_loader.get_next_hlc()
            if hlc != 0:
                if self.prev_hlc == 0:
                    self.prev_yaw = self.vehicle.get_transform().rotation.yaw
                else:
                    cur_yaw = self.vehicle.get_transform().rotation.yaw
                    self.delta_yaw += calculate_delta_yaw(self.prev_yaw, cur_yaw)
                    self.prev_yaw = cur_yaw
            
            # detect whether the vehicle made the correct turn
            turning_infraction = False
            if self.prev_hlc != 0 and hlc == 0:
                print(f'turned {self.delta_yaw} degrees')
                if 45 < np.abs(self.delta_yaw) < 180:
                    if self.delta_yaw < 0 and self.prev_hlc != 1:
                        turning_infraction = True
                    elif self.delta_yaw > 0 and self.prev_hlc != 2:
                        turning_infraction = True
                elif self.prev_hlc != 3:
                    turning_infraction = True
                if turning_infraction:
                    print('Wrong Turn!!!')
                    self.termination_code = 2
                self.delta_yaw = 0
            
            self.prev_hlc = hlc

            # get traffic light status
            light_status = -1
            vehicle_location = self.vehicle.get_transform().location
            if self.vehicle.is_at_traffic_light():
                traffic_light = self.vehicle.get_traffic_light()
                light_status = traffic_light.get_state()
                traffic_light_location = traffic_light.get_transform().location
                distance_to_traffic_light = np.sqrt((vehicle_location.x - traffic_light_location.x)**2 + (vehicle_location.y - traffic_light_location.y)**2)
                if light_status == carla.libcarla.TrafficLightState.Red and distance_to_traffic_light < 6 and speed_m_s > 5:
                    if not self.running_light:
                        self.running_light = True
                        self.red_light_counter += 1
                else:
                    self.running_light = False

            print(f'high-level command: {hlc}')
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

            # calculate distance to target point
            # print(f'vehicle location: ({vehicle_location.x}, {-vehicle_location.y})')
            # print(f'target point: ({self.target_point[0]}, {self.target_point[1]})')
            if self.target_point != None:
                distance_to_target = np.sqrt((self.target_point[0] - vehicle_location.x)**2 + (self.target_point[1] - (-vehicle_location.y))**2)
                print(f'Euclidean distance to target: {distance_to_target}')
                if distance_to_target < 1.5:
                    self.termination_code = 1


        except Exception as err:
            print(err)