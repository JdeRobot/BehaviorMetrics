from brains.CARLA.TCP.model import TCP
from brains.CARLA.TCP.config import GlobalConfig
from collections import OrderedDict

from brains.CARLA.utils.test_utils import traffic_light_to_int, model_control, calculate_delta_yaw
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from brains.CARLA.utils.high_level_command import HighLevelCommandLoader
from os import path

import numpy as np

import torch
import time
import math
import carla

from utils.logger import logger
import importlib
from torchvision import transforms as T
from brains.CARLA.TCP.leaderboard.team_code.planner import RoutePlanner
from brains.CARLA.TCP.leaderboard.leaderboard.utils.route_manipulation import downsample_route
from brains.CARLA.TCP.leaderboard.leaderboard.utils.route_manipulation import interpolate_trajectory
from brains.CARLA.TCP.leaderboard.leaderboard.utils.route_indexer import RouteIndexer

from collections import deque


PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'CARLA/'

class Brain:

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera_rgb = sensors.get_camera('camera_0') # rgb front view camera
        self.camera_seg = sensors.get_camera('camera_2') # segmentation camera
        self.imu = sensors.get_imu('imu_0') # imu
        self.gnss = sensors.get_gnss('gnss_0') # gnss
        self.speedometer = sensors.get_speedometer('speedometer_0') # gnss
        self.handler = handler
        self.inference_times = []
        self.gpu_inference = config['GPU']
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.gpu_inference) else 'cpu')

        client = carla.Client('localhost', 2000)
        client.set_timeout(100.0)
        world = client.get_world()
        self.map = world.get_map()

        world.set_weather(carla.WeatherParameters.ClearNoon)

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            
            else:
                self.config = GlobalConfig()
                self.net = TCP(self.config).to(self.device)

                ckpt = torch.load(PRETRAINED_MODELS + model,map_location=self.device)
                ckpt = ckpt["state_dict"]
                new_state_dict = OrderedDict()
                for key, value in ckpt.items():
                    new_key = key.replace("model.","")
                    new_state_dict[new_key] = value
                self.net.load_state_dict(new_state_dict, strict = False)

                #self.net.load_state_dict(torch.load(PRETRAINED_MODELS + model,map_location=self.device))
                self.net.cuda()
                self.net.eval()

        self.vehicle = None
        while self.vehicle is None:
            for vehicle in world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == 'ego_vehicle':
                    self.vehicle = vehicle
                    break
            if self.vehicle is None:
                print("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again

        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])


        repetitions = 1
        routes = 'brains/CARLA/TCP/leaderboard/data/TCP_training_routes/routes_town02.xml'
        scenarios = 'brains/CARLA/TCP/leaderboard/data/scenarios/town02_all_scenarios.json'
        route_indexer = RouteIndexer(routes, scenarios, repetitions)
        # setup
        config = route_indexer.next()
        '''
        WORKS WITH:
        result['target_point'][0] *= -1
        self.target_point = torch.stack([result['target_point'][1], result['target_point'][0]], dim=1).to('cuda', dtype=torch.float32)
        '''
        config.trajectory[0].x = 30.3
        config.trajectory[0].y = 109.5

        config.trajectory[1].x = 80.0
        config.trajectory[1].y = 109.5
        
        '''
        WORKS WITH:
        self.target_point = torch.stack([result['target_point'][1], result['target_point'][0]], dim=1).to('cuda', dtype=torch.float32)
        '''
        #config.trajectory[0].x = 55.3
        #config.trajectory[0].y = -105.6

        #config.trajectory[1].x = -30.0
        #config.trajectory[1].y = -105.6

        # WORKS with result['target_point'][1] *= -1
        '''
        WORKS WITH:
        result['target_point'][1] *= -1
        self.target_point = torch.stack(result['target_point'], dim=1).to('cuda', dtype=torch.float32)
        '''
        #config.trajectory[0].x = -3.3
        #config.trajectory[0].y = 179.5
        
        #config.trajectory[1].x = -3.3
        #config.trajectory[1].y = 120.6

        # WORKS without result['target_point'][1] *= -1
        '''
        WORKS WITH:
        #result['target_point'][1] *= -1
        self.target_point = torch.stack(result['target_point'], dim=1).to('cuda', dtype=torch.float32)
        '''
        #config.trajectory[0].x = -7.43
        #config.trajectory[0].y = 125.5
        
        #config.trajectory[1].x = -7.43
        #config.trajectory[1].y = 170.6

        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(world, config.trajectory)

        self.route = route
        self.set_global_plan(gps_route, self.route)
        self._init_route_planner()

        self.steer_step = 0
        self.last_steers = deque()
        self.status = 0


    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp=None):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]


    def _init_route_planner(self):
        self._route_planner = RoutePlanner(4.0, 50.0) # min_distance, max_distance
        self._route_planner.set_route(self._global_plan, True)

        gps = np.array([self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']])
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        self.initialized = True


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


    def _get_position(self, tick_data):
        gps = self.gnss.getGNSS()
        gps = np.array([gps.longitude, gps.latitude])
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    @torch.no_grad()
    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        # VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
        command = 4
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)

        rgb = self.camera_rgb.getImage().data
        seg_image = self.camera_seg.getImage().data
        
        self.update_frame('frame_0', rgb)
        self.update_frame('frame_1', seg_image)


        speed = self.speedometer.getSpeedometer().data
        gt_velocity = torch.FloatTensor([speed]).to('cuda', dtype=torch.float32)
        speed = torch.tensor(speed).view(-1, 1).to('cuda', dtype=torch.float32)
        speed = speed / 12

        imu_data = self.imu.getIMU()
        compass = np.array([imu_data.compass.x, imu_data.compass.y, imu_data.compass.z])
        compass = compass[-1]

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0

        result = {
                'rgb': rgb,
                #'gps': gps,
                'speed': speed,
                'compass': compass,
                #'bev': bev
                }

        pos = self._get_position(result)
        #result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        next_wp = [next_wp[1], next_wp[0]]
        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])



        result['target_point'] = tuple(local_command_point)
        result['target_point'] = [torch.FloatTensor([result['target_point'][0]]),
                                        torch.FloatTensor([result['target_point'][1]])]
        #result['target_point'][1] *= -1
        self.target_point = torch.stack(result['target_point'], dim=1).to('cuda', dtype=torch.float32)
        #self.target_point = torch.stack([result['target_point'][1], result['target_point'][0]], dim=1).to('cuda', dtype=torch.float32)

        state = torch.cat([speed, self.target_point, cmd_one_hot], 1)

        rgb = self._im_transform(rgb).unsqueeze(0).to('cuda', dtype=torch.float32)
        pred = self.net(rgb, state, self.target_point)

        
        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, command, gt_velocity, self.target_point)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, self.target_point)

        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0


        
        if self.status == 0:
            self.alpha = 0.3
            steer_ctrl = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
            throttle_ctrl = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
            brake_ctrl = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            steer_ctrl = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
            throttle_ctrl = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
            brake_ctrl = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)

        if brake_ctrl > 0.5:
            throttle_ctrl = float(0)


        self.motors.sendThrottle(throttle_ctrl)
        self.motors.sendSteer(steer_ctrl)
        self.motors.sendBrake(brake_ctrl)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(steer_ctrl)))

        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1

        else:
            self.status = 0
        
