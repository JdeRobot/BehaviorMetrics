#!/usr/bin/env python
""" This module is responsible for handling the sensors of the robot.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
from utils.logger import logger

# Import ROS only if needed
ROS_VERSION = os.environ.get('ROS_VERSION')
USE_ROS = ROS_VERSION in ('1', '2')

if USE_ROS:
    from robot.interfaces.camera import ListenerCamera
    from robot.interfaces.laser import ListenerLaser
    from robot.interfaces.pose3d import ListenerPose3d    
    from robot.interfaces.speedometer import ListenerSpeedometer
else:
    # CARLA PYTHON API implementations of the sensors
    from robot.interfaces.carla_api_sensors import (
        CarlaApiCamera, CarlaApiLidar, CarlaApiPose3D, CarlaApiSpeedometer
    )
try:
    from robot.interfaces.birdeyeview import BirdEyeView
except ModuleNotFoundError as ex:
    logger.error('CARLA is not supported sensor')

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Sensors:
    """This class controls the creation of the actuators of the robot

    Attributes:
        cameras {dict} -- Dictionary which key is the name of the motor and value is a camera instance.
        laser {dict} -- Dictionary which key is the name of the motor and value is a laser sensor instance.
        pose3d {dict} -- Dictionary which key is the name of the motor and value is an odometry instance.
    """

    def __init__(self, sensors_config, node):
        """Constructor of the class

        Arguments:
            sensors_config {dict} -- Configuration of the different sensors.
        """
        self.node = node

        # Load cameras
        cameras_conf = sensors_config.get('Cameras', None)
        self.cameras = None
        if cameras_conf:
            self.cameras = self.__create_sensor(cameras_conf, 'camera')

        # Load lasers
        lasers_conf = sensors_config.get('Lasers', None)
        self.lasers = None
        if lasers_conf:
            self.lasers = self.__create_sensor(lasers_conf, 'laser')

        # Load pose3d
        pose3d_conf = sensors_config.get('Pose3D', None)
        if pose3d_conf:
            self.pose3d = self.__create_sensor(pose3d_conf, 'pose3d')

        # Load BirdEyeView
        bird_eye_view_conf = sensors_config.get('BirdEyeView', None)
        self.bird_eye_view = None
        if bird_eye_view_conf:
            self.bird_eye_view = self.__create_sensor(bird_eye_view_conf, 'bird_eye_view')

        # Load speedometer
        speedometer_conf = sensors_config.get('Speedometer', None)
        self.speedometer = None
        if speedometer_conf:
            self.speedometer = self.__create_sensor(speedometer_conf, 'speedometer')

    def __create_sensor(self, sensor_config, sensor_type):
        """Fill the sensor dictionary with instances of the sensor_type and sensor_config"""
        sensor_dict = {}
        for elem, cfg in sensor_config.items():
            name = cfg['Name']
            backend = cfg.get('Backend', 'ros' if USE_ROS else 'python_api').lower()
            
            if backend not in ('ros', 'python_api', 'carla_api'):
                raise ValueError(f"Unsupported backend '{backend}' for sensor '{name}'")
            
            if backend == 'ros':
                if not USE_ROS:
                    raise RuntimeError(f"ROS backend requested for sensor '{name}', but ROS is not available.")
            
                topic = cfg['Topic']
                if sensor_type == 'camera':
                    sensor_dict[name] = ListenerCamera(self.node, topic)
                elif sensor_type == 'laser':
                    sensor_dict[name] = ListenerLaser(self.node, topic)
                elif sensor_type == 'pose3d':
                    sensor_dict[name] = ListenerPose3d(self.node, topic)
                elif sensor_type == 'bird_eye_view':
                    sensor_dict[name] = BirdEyeView()
                elif sensor_type == 'speedometer':
                    sensor_dict[name] = ListenerSpeedometer(self.node, topic)
                
            else:
                # Python api backends
                if sensor_type == 'camera':
                    sensor_dict[name] = CarlaApiCamera(cfg)
                elif sensor_type == 'laser':
                    sensor_dict[name] = CarlaApiLidar(cfg)
                elif sensor_type == 'pose3d':
                    sensor_dict[name] = CarlaApiPose3D(cfg)
                # elif sensor_type == 'bird_eye_view':
                #     sensor_dict[name] = BirdEyeView()
                elif sensor_type == 'speedometer':
                    sensor_dict[name] = CarlaApiSpeedometer(cfg)

        return sensor_dict

    def __get_sensor(self, sensor_name, sensor_type):
        """Retrieve an specific sensor"""

        sensor = None
        try:
            if sensor_type == 'camera':
                sensor = self.cameras[sensor_name]
            elif sensor_type == 'laser':
                sensor = self.lasers[sensor_name]
            elif sensor_type == 'pose3d':
                sensor = self.pose3d[sensor_name]
            elif sensor_type == 'bird_eye_view':
                sensor = self.bird_eye_view[sensor_name]
            elif sensor_type == 'speedometer':
                sensor = self.speedometer[sensor_name]
        except KeyError:
            return "[ERROR] No existing camera with {} name.".format(sensor_name)

        return sensor

    def get_camera(self, camera_name):
        """Retrieve an specific existing camera

        Arguments:
            camera_name {str} -- Name of the camera to be retrieved

        Returns:
            robot.interfaces.camera.ListenerCamera instance -- camera instance
        """
        return self.__get_sensor(camera_name, 'camera')

    def get_laser(self, laser_name):
        """Retrieve an specific existing laser

        Arguments:
            laser_name {str} -- Name of the laser to be retrieved

        Returns:
            robot.interfaces.laser.ListenerLaser instance -- laser instance
        """
        return self.__get_sensor(laser_name, 'laser')

    def get_pose3d(self, pose_name):
        """Retrieve an specific existing pose3d sensor

        Arguments:
            pose_name {str} -- Name of the pose3d to be retrieved

        Returns:
            robot.interfaces.pose3d.ListenerPose3d instance -- pose3d instance
        """
        return self.__get_sensor(pose_name, 'pose3d')

    def get_bird_eye_view(self, bird_eye_view_name):
        """Retrieve an specific existing bird eye view

        Arguments:
            bird_eye_view_name {str} -- Name of the birdeyeview to be retrieved

        Returns:
            robot.interfaces.birdeyeview.BirdEyeView instance -- birdeyeview instance
        """
        return self.__get_sensor(bird_eye_view_name, 'bird_eye_view')

    def kill(self):
        """Destroy all the running sensors"""
        if self.cameras:
            for camera in self.cameras.values():
                camera.stop()
        if self.lasers:
            for laser in self.lasers.values():
                laser.stop()
        if self.pose3d:
            for pose in self.pose3d.values():
                pose.stop()
