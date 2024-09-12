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
        client.set_timeout(10.0)
        world = client.get_world()
        self.map = world.get_map()

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            
            else:
                # Initialize TCP variables
                self.alpha = 0.3
                self.status = 0
                self.steer_step = 0
                self.last_steers = deque()
                self.step = -1
                self.initialized = False
                self.config = GlobalConfig()
                self.net = TCP(self.config).to(self.device)

                ckpt = torch.load(PRETRAINED_MODELS + model,map_location=self.device)
                ckpt = ckpt["state_dict"]
                new_state_dict = OrderedDict() # TODO: Why an OrderedDict
                for key, value in ckpt.items():
                    new_key = key.replace("model.","")
                    new_state_dict[new_key] = value
                self.net.load_state_dict(new_state_dict, strict = False)
                self.net.cuda()
                self.net.eval()

                self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

        self.vehicle = None
        while self.vehicle is None:
            for vehicle in world.get_actors().filter('vehicle.*'):
                if vehicle.attributes.get('role_name') == 'ego_vehicle':
                    self.vehicle = vehicle
                    break
            if self.vehicle is None:
                print("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again

        # Added code
        repetitions = 1
        routes = 'brains/CARLA/TCP/leaderboard/data/TCP_training_routes/routes_town02.xml'
        scenarios = 'brains/CARLA/TCP/leaderboard/data/scenarios/town02_all_scenarios.json'
        route_indexer = RouteIndexer(routes, scenarios, repetitions)
        config = route_indexer.next()
        config.trajectory[0].x = 55.3
        config.trajectory[0].y = -105.6

        #config.trajectory[1].y = -30.0
        #config.trajectory[1].x = -105.6


        gps_route, route = interpolate_trajectory(world, config.trajectory)
        # Minimum distance is 50m
        for route_point in route:
            print(route_point[0].location, route_point[0].rotation, route_point[1])

        self.set_global_plan(gps_route, route)
        # Added code
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        for route_point in self._route_planner.route:
            print(route_point)

        self.initialized = True


    def _get_position(self):
        gps = self.gnss.getGNSS()
        gps = np.array([gps.longitude, gps.latitude])
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def set_global_plan(self, global_plan_gps, global_plan_world_coord, wp=None):
        """
        Set the plan (route) for the agent
        """
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        # Custom update
        for x in ds_ids:
            global_plan_gps[x][0]['lat'], global_plan_gps[x][0]['lon'] = global_plan_gps[x][0]['lon'], global_plan_gps[x][0]['lat']
    
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

    def tick(self):
        rgb = self.camera_rgb.getImage().data
        self.update_frame('frame_0', rgb)
        speed = self.speedometer.getSpeedometer().data
        #velocity = self.vehicle.get_velocity()
        #speed_m_s = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        #speed = 3.6 * speed_m_s #m/s to km/h 

        print('---SPEED 1---')
        print(speed)

        imu_data = self.imu.getIMU()

        compass = np.array([imu_data.compass.x, imu_data.compass.y, imu_data.compass.z, imu_data.compass.w])

        def convert_vector_to_compass_orientation(orientation_vector):
            _, _, orientation_x, orientation_y = orientation_vector

            compass_orientation = math.atan2(round(orientation_y, 2), round(orientation_x, 2))

            if compass_orientation < 0:
                compass_orientation += 2 * math.pi

            compass_orientation -= math.pi / 2.0

            return compass_orientation
        
        compass = convert_vector_to_compass_orientation(compass)

        
        result = {
				'rgb': rgb,
				#'gps': gps,
				'speed': speed,
				'compass': compass,
				#'bev': bev
				}
        
        pos = self._get_position()
        next_wp, next_cmd = self._route_planner.run_step(pos)

        result['next_command'] = next_cmd.value
        
        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])
        
        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        local_command_point[0], local_command_point[1] = local_command_point[1], -local_command_point[0]
        
        result['target_point'] = tuple(local_command_point)


        return result


    @torch.no_grad()
    def execute(self):

        tick_data = self.tick()
        if self.step < self.config.seq_len:
            #rgb = self._im_transform(tick_data['rgb']).unsqueeze(0)

            self.motors.sendThrottle(0.0)
            self.motors.sendSteer(0.0)
            self.motors.sendBrake(0.0)

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        cmd_one_hot = torch.tensor(cmd_one_hot).view(1, 6).to('cuda', dtype=torch.float32)
        speed = torch.FloatTensor([float(tick_data['speed'])]).view(1,1).to('cuda', dtype=torch.float32)
        speed = speed / 12
        print('---SPEED 2---')
        print(speed)


        print(tick_data['rgb'].shape)
        print(type(tick_data['rgb']))

        rgb = self._im_transform(tick_data['rgb']).unsqueeze(0).to('cuda', dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
										torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)
        state = torch.cat([speed, target_point, cmd_one_hot], 1)

        pred= self.net(rgb, state, target_point)

        steer_ctrl, throttle_ctrl, brake_ctrl, metadata = self.net.process_action(pred, tick_data['next_command'], gt_velocity, target_point)

        steer_traj, throttle_traj, brake_traj, metadata_traj = self.net.control_pid(pred['pred_wp'], gt_velocity, target_point)
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        self.pid_metadata = metadata_traj

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            steer = np.clip(self.alpha*steer_ctrl + (1-self.alpha)*steer_traj, -1, 1)
            throttle = np.clip(self.alpha*throttle_ctrl + (1-self.alpha)*throttle_traj, 0, 0.75)
            brake = np.clip(self.alpha*brake_ctrl + (1-self.alpha)*brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            steer = np.clip(self.alpha*steer_traj + (1-self.alpha)*steer_ctrl, -1, 1)
            throttle = np.clip(self.alpha*throttle_traj + (1-self.alpha)*throttle_ctrl, 0, 0.75)
            brake = np.clip(self.alpha*brake_traj + (1-self.alpha)*brake_ctrl, 0, 1)

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if brake > 0.5:
            throttle = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(steer)))

        num = 0
        for s in self.last_steers:
            if s > 0.10:
                num += 1
        if num > 10:
            self.status = 1
            self.steer_step += 1
        else:
            self.status = 0

        self.pid_metadata['status'] = self.status


        print(throttle, steer, brake)
        self.motors.sendThrottle(throttle)
        self.motors.sendSteer(steer)
        self.motors.sendBrake(brake)

        #self.motors.sendThrottle(1.0)
        #self.motors.sendSteer(0.0)
        #self.motors.sendBrake(0.0)



        '''
            TODO: REVIEW PID CONTROLLERS SINCE THEY USE A WINDOW
            TODO: Draw the waypoints on the map
            TODO: Draw the trajectory on the images.
            TODO: Figure out what's the transformation for the compass.
            TODO: Review metadata from model!!! metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}
        '''