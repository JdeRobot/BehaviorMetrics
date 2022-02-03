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
import rospy
import subprocess

from datetime import datetime
from brains.brains_handler import Brains
from robot.actuators import Actuators
from robot.sensors import Sensors
from utils.logger import logger
from utils.constants import MIN_EXPERIMENT_PERCENTAGE_COMPLETED
from rosgraph_msgs.msg import Clock

import numpy as np

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'



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
        self.robot_type = self.brain_path.split("/")[-2]
        self.sensors = None
        self.actuators = None
        self.brains = None
        self.initialize_robot()
        if self.robot_type == 'drone':
            self.pose3d = self.brains.active_brain.getPose3d()
            self.start_pose = np.array([self.pose3d[0], self.pose3d[1]])
        else:
            self.pose3d = self.sensors.get_pose3d('pose3d_0')
            self.start_pose = np.array([self.pose3d.getPose3d().x, self.pose3d.getPose3d().y])
        self.previous = datetime.now()
        self.checkpoints = []
        self.metrics = {}
        self.checkpoint_save = False
        self.max_distance = 0.5
        self.execution_completed = False
        self.stats_thread = threading.Thread(target=self.track_stats)
        self.stats_thread.start()
        self.ros_clock_time = 0
        self.real_time_factor = 0
        self.brain_iterations_real_time = []
        self.brain_iterations_simulated_time = []
        self.real_time_factors = []
        self.real_time_update_rate = 1000
        self.pilot_start_time = 0
        self.time_cycle = self.configuration.pilot_time_cycle

    def __wait_gazebo(self):
        """Wait for gazebo to be initialized"""

        # gazebo_ready = False
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
        if self.robot_type != 'drone':
            self.actuators = Actuators(self.configuration.actuators)
            self.sensors = Sensors(self.configuration.sensors)
        if hasattr(self.configuration, 'experiment_model') and type(self.configuration.experiment_model) != list:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller,
                                 self.configuration.experiment_model, self.configuration.brain_kwargs)
        else:
            self.brains = Brains(self.sensors, self.actuators, self.brain_path, self.controller,
                                 config=self.configuration.brain_kwargs)
        self.__wait_gazebo()

    def stop_interfaces(self):
        """Function that kill the current interfaces of the robot. For reloading purposes."""
        if self.sensors:
            self.sensors.kill()
        if self.actuators:
            self.actuators.kill()
        pass

    def run(self):
        """Main loop of the class. Calls a brain action every self.time_cycle"""
        "TODO: cleanup measure of ips"
        it = 0
        ss = time.time()
        self.brain_iterations_real_time = []
        self.brain_iterations_simulated_time = []
        self.real_time_factors = []
        self.sensors.get_camera('camera_0').total_frames = 0
        self.pilot_start_time = time.time()
        while not self.kill_event.is_set():
            if not self.stop_event.is_set():
                start_time = datetime.now()
                start_time_ros = self.ros_clock_time
                self.execution_completed = False
                try:
                    self.brains.active_brain.execute()
                except AttributeError as e:
                    logger.warning('No Brain selected')
                    logger.error(e)

                dt = datetime.now() - start_time
                ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
                self.brain_iterations_real_time.append(ms / 1000)
                elapsed = time.time() - ss
                if elapsed < 1:
                    it += 1
                else:
                    ss = time.time()
                    it = 0
                if ms < self.time_cycle:
                    time.sleep((self.time_cycle - ms) / 1000.0)
                self.real_time_factors.append(self.real_time_factor)
                self.brain_iterations_simulated_time.append(self.ros_clock_time - start_time_ros)
        self.execution_completed = True
        self.clock_subscriber.unregister()
        self.stats_process.terminate()
        poll = self.stats_process.poll()
        while poll is None:
            time.sleep(1)
            poll = self.stats_process.poll()
        self.kill()
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
        self.stop_interfaces()
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
        if dist < self.max_distance:
            return True
        return False

    def calculate_metrics(self, experiment_metrics):
        if hasattr(self.brains.active_brain, 'inference_times'):
            self.brains.active_brain.inference_times = self.brains.active_brain.inference_times[10:-10]
            mean_inference_time = sum(self.brains.active_brain.inference_times) / len(self.brains.active_brain.inference_times)
            frame_rate = self.sensors.get_camera('camera_0').total_frames / experiment_metrics['experiment_total_simulated_time']
            gpu_inference = self.brains.active_brain.gpu_inference
            real_time_update_rate = self.real_time_update_rate
            first_image = self.brains.active_brain.first_image
            logger.info('* Mean network inference time ---> ' + str(mean_inference_time) + 's')
            logger.info('* Frame rate ---> ' + str(frame_rate) + 'fps')
        else:
            mean_inference_time = 0
            frame_rate = 0
            gpu_inference = False
            real_time_update_rate = self.real_time_update_rate
            first_image = None
            logger.info('No deep learning based brain')
        if self.brain_iterations_real_time and self.brain_iterations_simulated_time and self.brain_iterations_simulated_time:
            mean_brain_iteration_simulated_time = sum(self.brain_iterations_simulated_time) / len(self.brain_iterations_simulated_time)
            real_time_factor = sum(self.real_time_factors) / len(self.real_time_factors)
            brain_iterations_frequency_simulated_time = 1 / mean_brain_iteration_simulated_time
            target_brain_iteration_simulated_time = 1000 / self.time_cycle / round(real_time_factor, 1)
            mean_brain_iteration_real_time = sum(self.brain_iterations_real_time) / len(self.brain_iterations_real_time)
            brain_iterations_frequency_real_time = 1 / mean_brain_iteration_real_time
            target_brain_iteration_real_time = 1 / (self.time_cycle / 1000)
        else:
            mean_brain_iteration_real_time = 0
            mean_brain_iteration_simulated_time = 0
            real_time_factor = 0
            brain_iterations_frequency_simulated_time = 0
            target_brain_iteration_simulated_time = 0
            brain_iterations_frequency_real_time = 0
            target_brain_iteration_real_time = 0
        logger.info('* Brain iterations frequency simulated time ---> ' + str(brain_iterations_frequency_simulated_time) + 'it/s')
        logger.info('* Target brain iteration simulated time -> ' + str(target_brain_iteration_simulated_time) + 'it/s')
        logger.info('* Mean brain iteration real time ---> ' + str(mean_brain_iteration_real_time) + 's')
        logger.info('* Brain iterations frequency real time ---> ' + str(brain_iterations_frequency_real_time) + 'it/s')
        logger.info('* Target brain iteration real time -> ' + str(target_brain_iteration_real_time) + 'it/s')
        logger.info('* ;ean brain iteration simulated time ---> ' + str(mean_brain_iteration_simulated_time) + 's')
        logger.info('* Mean brain iteration simulated time ---> ' + str(real_time_factor))
        logger.info('* Real time update rate ---> ' + str(real_time_update_rate))
        logger.info('* GPU inference ---> ' + str(gpu_inference))
        logger.info('* Saving experiment ---> ' + str(hasattr(self.controller, 'experiment_metrics_filename')))
        experiment_metrics['brain_iterations_frequency_simulated_time'] = brain_iterations_frequency_simulated_time
        experiment_metrics['target_brain_iteration_simulated_time'] = target_brain_iteration_simulated_time
        experiment_metrics['mean_brain_iteration_real_time'] = mean_brain_iteration_real_time
        experiment_metrics['brain_iterations_frequency_real_time'] = brain_iterations_frequency_real_time
        experiment_metrics['target_brain_iteration_real_time'] = target_brain_iteration_real_time
        experiment_metrics['mean_inference_time'] = mean_inference_time
        experiment_metrics['frame_rate'] = frame_rate
        experiment_metrics['gpu_inference'] = gpu_inference
        experiment_metrics['mean_brain_iteration_simulated_time'] = mean_brain_iteration_simulated_time
        experiment_metrics['real_time_factor'] = real_time_factor
        experiment_metrics['real_time_update_rate'] = real_time_update_rate
        logger.info('Saving metrics to ROS bag')
        return experiment_metrics, first_image

    def clock_callback(self, clock_data):
        self.ros_clock_time = clock_data.clock.to_sec()

    def track_stats(self):
        args = ["gz", "stats", "-p"]
        # Prints gz statistics. "-p": Output comma-separated values containing-
        # real-time factor (percent), simtime (sec), realtime (sec), paused (T or F)
        self.stats_process = subprocess.Popen(args, stdout=subprocess.PIPE)
        # bufsize=1 enables line-bufferred mode (the input buffer is flushed
        # automatically on newlines if you would write to process.stdin )
        poll = self.stats_process.poll()
        while poll is not None:
            time.sleep(1)
            poll = self.stats_process.poll()

        self.clock_subscriber = rospy.Subscriber("/clock", Clock, self.clock_callback)
        with self.stats_process.stdout:
            for line in iter(self.stats_process.stdout.readline, b''):
                stats_list = [x.strip() for x in line.split(b',')]
                try:
                    self.real_time_factor = float(stats_list[0].decode("utf-8"))
                except Exception as ex:
                    self.real_time_factor = 0
