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

from .interfaces.motors import PublisherMotors

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Actuators:
    """This class controls the creation of the actuators of the robot

    Attributes:
        motors {dict} -- Dictionary which key is the name of the motor and value is a ROS motors publisher instance.

    """

    def __init__(self, actuators_config):
        """Constructor of the class

        Arguments:
            actuators_config {dict} -- Configuration of the different actuators.
        """

        # Load motors
        motors_conf = actuators_config.get('Motors', None)
        self.motors = None
        if motors_conf:
            self.motors = self.__create_actuator(motors_conf, 'motor')

    def __create_actuator(self, actuator_config, actuator_type):
        """Fill the motors dictionary with instances of the motors to control the robot"""

        actuator_dict = {}
        for elem in actuator_config:
            name = actuator_config[elem]['Name']
            topic = actuator_config[elem]['Topic']
            vmax = actuator_config[elem]['MaxV']
            wmax = actuator_config[elem]['MaxW']
            
            if 'RL' in actuator_config[elem]:
                if actuator_config[elem]['RL'] == False:
                    if actuator_type == 'motor':
                        actuator_dict[name] = PublisherMotors(topic, vmax, wmax, 0, 0)
            else:
                if actuator_type == 'motor':
                        actuator_dict[name] = PublisherMotors(topic, vmax, wmax, 0, 0)
        return actuator_dict

    def __get_actuator(self, actuator_name, actuator_type):
        """Retrieve an specific actuator"""

        actuator = None
        try:
            if actuator_type == 'motor':
                actuator = self.motors[actuator_name]
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
        for actuator in self.motors.values():
            actuator.stop()
