#!/usr/bin/env python
""" This module is responsible for handling the actuators of the robot.

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


ROS_VERSION = os.environ.get("ROS_VERSION")
USE_ROS = ROS_VERSION in ("1", "2")

if USE_ROS:
    from .interfaces.motors import PublisherMotors, PublisherCARLAMotors
else:
    from .interfaces.carla_api_motors import CarlaApiMotors

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Actuators:
    """This class controls the creation of the actuators of the robot

    Attributes:
        motors {dict} -- Dictionary which key is the name of the motor and value is a ROS motors publisher instance.

    """

    def __init__(self, actuators_config, node):
        """Constructor of the class

        Arguments:
            actuators_config {dict} -- Configuration of the different actuators.
        """
        self.node = node
        # Load motors
        motors_conf = actuators_config.get('Motors', None)
        carla_motors_conf = actuators_config.get('CARLA_Motors', None)
        self.motors = None
        if motors_conf:
            self.motors = self.__create_actuator(motors_conf, 'motor')
        elif carla_motors_conf:
            self.motors = self.__create_actuator(carla_motors_conf, 'carla_motor')

    def __create_actuator(self, actuator_config, actuator_type):
        """Fill the motors dictionary with instances of the motors to control the robot"""

        actuator_dict = {}
        for elem, cfg in actuator_config.items():
            cfg = actuator_config[elem]
            name = cfg['Name']
            vmax = cfg['MaxV']
            wmax = cfg['MaxW']
 
            if actuator_type == 'motor':
                topic = cfg['Topic']
                actuator_dict[name] = PublisherMotors(self.node, topic, vmax, wmax, 0, 0)

            elif actuator_type == 'carla_motor':
                if not USE_ROS:
                    actuator_dict[name] = CarlaApiMotors(vmax, wmax)
                else:
                    topic = cfg.get('Topic', '')  # ROS 
                    if not topic:
                        raise ValueError("Actuators: 'Topic' no puede estar vacío para backend ROS.")
                    actuator_dict[name] = PublisherCARLAMotors(self.node, topic, vmax, wmax, 0, 0)
        return actuator_dict

    def __get_actuator(self, actuator_name, actuator_type):
        """Retrieve an specific actuator"""

        try:
            return self.motors[actuator_name]
        except KeyError:
            return "[ERROR] No existing actuator with {} name.".format(actuator_name)

        return actuator

    def get_motor(self, motor_name):
        """Retrieve an specific existing motor

        Arguments:
            motor_name {str} -- Name of the motor to be retrieved

        Returns:
            robot.interfaces.motors.PublisherMotors instance -- ROS motor instance
        """
        return self.__get_actuator(motor_name, 'motor')

    def kill(self):
        """Destroy all the running actuators"""
        # do the same for every publisher that requires threading
        if not self.motors:
            return
        for actuator in self.motors.values():
            if hasattr(actuator, "destroy"):
                try: actuator.destroy()
                except Exception: pass
            elif hasattr(actuator, "stop"):
                try: actuator.stop()
                except Exception: pass
