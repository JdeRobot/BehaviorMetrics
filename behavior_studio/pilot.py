#!/usr/bin/env python
""" This module is responsible for handling the logic of the robot and its current brain.

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

import threading
import time
from datetime import datetime

import Pyro4

from brains.brains_handler import Brains
from robot.actuators import Actuators
from robot.sensors import Sensors
from utils.logger import logger

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

TIME_CYCLE = 60

@Pyro4.expose
class Pilot(threading.Thread):
    """This class handles the robot and its brain.

    This class called Pilot that handles the initialization of the robot sensors and actuators and the
    brain that will control the robot. The main logic consists of an infinite loop called every 60 milliseconds that
    invoke an action from the brain.

    Attributes:
        controller {utils.controller.Controller} -- Controller instance of the MVC of the application
        configuration {utils.configuration.Config} -- Configuration instance of the application
        sensors {robot.sensors.Sensors} -- Sensors instance of the robot
        actuators {robot.actuators.Actuators} -- Actuators instance of the robot
        brains {brains.brains_handler.Brains} -- Brains controller instance
    """

    def __init__(self, configuration, controller, headless=False):
        """Constructor of the pilot class

        Arguments:
            configuration {utils.configuration.Config} -- Configuration instance of the application
            controller {utils.controller.Controller} -- Controller instance of the MVC of the application
        """
        self.controller = controller
        self.controller.set_pilot(self)
        self.configuration = configuration
        self.headless = headless

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        threading.Thread.__init__(self, args=self.stop_event)

        self.sensors = None
        self.actuators = None
        self.brains = None
        self.initialize_robot()

    def __wait_gazebo(self):
        """Wait for gazebo to be initialized"""

        gazebo_ready = False
        self.stop_event.set()
        while not gazebo_ready:
            try:
                self.controller.pause_gazebo_simulation()
                gazebo_ready = True
                self.stop_event.clear()
            except Exception as ex:
                print(ex)

    def initialize_robot(self):
        """Initialize robot interfaces (sensors and actuators) and its brain from configuration"""

        self.stop_interfaces()
        self.actuators = Actuators(self.configuration.actuators)
        self.sensors = Sensors(self.configuration.sensors)
        self.brains = Brains(self.sensors, self.actuators, self.configuration.brain_path, self.controller)
        # if not self.headless:
        self.__wait_gazebo()

    def stop_interfaces(self):
        """Function that kill the current interfaces of the robot. For reloading purposes."""

        if self.sensors:
            self.sensors.kill()
        if self.actuators:
            self.actuators.kill()
        pass

    def run(self):
        """Main loop of the class. Calls a brain action every TIME_CYCLE"""
        "TODO: cleanup measure of ips"
        it = 0
        ss = time.time()
        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                try:
                    self.brains.active_brain.execute()
                except AttributeError as e:
                    logger.warning('No Brain selected')
                    logger.error(e)

            dt = datetime.now() - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            elapsed = time.time() - ss
            if elapsed < 1:
                it += 1
            else:
                ss = time.time()
                # print(it)
                it = 0

            if (ms < TIME_CYCLE):
                time.sleep((TIME_CYCLE - ms) / 1000.0)
        logger.debug('Pilot: pilot killed.')

    def stop(self):
        """Pause the main loop"""

        self.stop_event.set()

    def play(self):
        """Resume the main loop."""

        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        """Destroy the main loop. For exiting"""

        self.actuators.kill()
        self.kill_event.set()

    def reload_brain(self, brain_path):
        """Reload a brain specified by brain_path

        This function is useful if one wants to change the environment of the robot (simulated world).

        Arguments:
            brain_path {str} -- Path to the brain module to load.
        """
        self.brains.load_brain(brain_path)
