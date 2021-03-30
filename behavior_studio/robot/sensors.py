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

from robot.interfaces.camera import ListenerCamera
from robot.interfaces.laser import ListenerLaser
from robot.interfaces.pose3d import ListenerPose3d

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

    def __init__(self, sensors_config):
        """Constructor of the class

        Arguments:
            sensors_config {dict} -- Configuration of the different sensors.
        """

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

    def __create_sensor(self, sensor_config, sensor_type):
        """Fill the sensor dictionary with instances of the sensor_type and sensor_config"""
        sensor_dict = {}
        for elem in sensor_config:
            name = sensor_config[elem]['Name']
            topic = sensor_config[elem]['Topic']
            if sensor_type == 'camera':
                sensor_dict[name] = ListenerCamera(topic)
            elif sensor_type == 'laser':
                sensor_dict[name] = ListenerLaser(topic)
            elif sensor_type == 'pose3d':
                sensor_dict[name] = ListenerPose3d(topic)

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
