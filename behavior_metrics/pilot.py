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

from brains.brains_handler import Brains
from robot.actuators import Actuators
from robot.sensors import Sensors
from utils.logger import logger
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED

import numpy as np

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

TIME_CYCLE = 50


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

    def __init__(self, configuration, controller, brain_path):
        """Constructor of the pilot class

        Arguments:
            configuration {utils.configuration.Config} -- Configuration instance of the application
            controller {utils.controller.Controller} -- Controller instance of the MVC of the application
        """
        
        self.controller = controller
        self.controller.set_pilot(self)
        self.configuration = configuration
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        threading.Thread.__init__(self, args=self.stop_event)
        self.brain_path = brain_path
        self.sensors = None
        self.actuators = None
        self.brains = None
        self.initialize_robot()
        self.pose3d = self.sensors.get_pose3d('pose3d_0')
        self.start_pose = np.array([self.pose3d.getPose3d().x, self.pose3d.getPose3d().y])
        self.previous = datetime.now()
        self.checkpoints = []
        self.metrics = {}
        self.checkpoint_save = False
        self.max_distance = 0.5
        

    def __wait_gazebo(self):
        """Wait for gazebo to be initialized"""

        gazebo_ready = False
        self.stop_event.set()
#         while not gazebo_ready:
#             try:
#                 self.controller.pause_gazebo_simulation()
#                 gazebo_ready = True
#                 self.stop_event.clear()
#             except Exception as ex:
#                 print(ex)

    def initialize_robot(self):
        """Initialize robot interfaces (sensors and actuators) and its brain from configuration"""
        self.stop_interfaces()
        self.actuators = Actuators(self.configuration.actuators)
        self.sensors = Sensors(self.configuration.sensors)
        if hasattr(self.configuration, 'experiment_model'):
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller, self.configuration.experiment_model)
        elif hasattr(self.configuration, 'experiment_model') and type(self.configuration.experiment_model) != list:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller, self.configuration.experiment_model)
        else:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller)
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
        stopped_brain_stats = False
        succesful_iteration = False
        brain_iterations_time = []
        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                stopped_brain_stats = True
                try:
                    self.brains.active_brain.execute()
                    succesful_iteration = True
                except AttributeError as e:
                    logger.warning('No Brain selected')
                    logger.error(e)
                    succesful_iteration = False
            else:
                if stopped_brain_stats:
                    stopped_brain_stats = False
                    succesful_iteration = False
                    try:
                        logger.info('----- MEAN INFERENCE TIME -----')
                        self.brains.active_brain.inference_times = self.brains.active_brain.inference_times[10:-10]
                        mean_inference_time = sum(self.brains.active_brain.inference_times) / len(self.brains.active_brain.inference_times)
                        frame_rate = len(self.brains.active_brain.inference_times) / sum(self.brains.active_brain.inference_times)
                        gpu_inferencing = self.brains.active_brain.gpu_inferencing
                        first_image = self.brains.active_brain.first_image
                        logger.info(mean_inference_time)
                        logger.info(frame_rate)
                        logger.info('-------------------')
                    except:
                        mean_inference_time = 0
                        frame_rate = 0
                        gpu_inferencing = False
                        first_image = 0
                        logger.info('No inference brain')
                    logger.info('----- MEAN ITERATION TIME -----')
                    mean_iteration_time = sum(brain_iterations_time) / len(brain_iterations_time)
                    logger.info(mean_iteration_time)
                    logger.info('-------------------')
                    if hasattr(self.controller, 'stats_filename') and self.controller.lap_statistics['percentage_completed'] > MIN_EXPERIMENT_PERCENTAGE_COMPLETED:
                        try:
                            self.controller.save_time_stats(mean_iteration_time, mean_inference_time, frame_rate, gpu_inferencing, first_image)
                        except:
                            logger.info('Empty ROS bag')
                    brain_iterations_time = [] 
            dt = datetime.now() - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if succesful_iteration:
                brain_iterations_time.append(ms/1000)
            elapsed = time.time() - ss
            if elapsed < 1:
                it += 1
            else:
                ss = time.time()
                it = 0

            if (ms < TIME_CYCLE):
                time.sleep((TIME_CYCLE - ms) / 1000.0)
        
        logger.info('Pilot: pilot killed.')

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

    def reload_brain(self, brain_path, model=None):
        """Reload a brain specified by brain_path

        This function is useful if one wants to change the environment of the robot (simulated world).

        Arguments:
            brain_path {str} -- Path to the brain module to load.
        """
        self.brains.load_brain(brain_path, model=model)
            
    def finish_line(self):
        pose = self.pose3d.getPose3d()
        current_point = np.array([pose.x, pose.y])

        dist = (self.start_pose - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        # print(dist)
        if dist < self.max_distance:
            return True
        return False
        
