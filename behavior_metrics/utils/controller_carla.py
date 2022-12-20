#!/usr/bin/env python

"""This module contains the controller of the application.

This application is based on a type of software architecture called Model View Controller. This is the controlling part
of this architecture (controller), which communicates the logical part (model) with the user interface (view).

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

import shlex
import subprocess
import threading
import cv2
import rospy
import os
import time
import rosbag
import json
import math
import carla

from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
from utils.logger import logger
from utils.constants import CIRCUITS_TIMEOUTS
from std_msgs.msg import String
from utils import metrics_carla
from carla_msgs.msg import CarlaLaneInvasionEvent
from carla_msgs.msg import CarlaCollisionEvent
from PIL import Image

__author__ = 'sergiopaniego'
__contributors__ = []
__license__ = 'GPLv3'


class ControllerCarla:
    """This class defines the controller of the architecture, responsible of the communication between the logic (model)
    and the user interface (view).

    Attributes:
        data {dict} -- Data to be sent to the view. The key is a frame_if of the view and the value is the data to be
        displayed. Depending on the type of data the frame handles (images, laser, etc)
        pose3D_data -- Pose data to be sent to the view
        recording {bool} -- Flag to determine if a rosbag is being recorded
    """

    def __init__(self):
        """ Constructor of the class. """
        pass
        self.__data_loc = threading.Lock()
        self.__pose_loc = threading.Lock()
        self.data = {}
        self.pose3D_data = None
        self.recording = False
        self.cvbridge = CvBridge()

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()

        self.carla_map = world.get_map()
        time.sleep(5)
        self.ego_vehicle = world.get_actors().filter('vehicle.*')[0]
        self.map_waypoints = self.carla_map.generate_waypoints(0.5)
        
    # GUI update
    def update_frame(self, frame_id, data):
        """Update the data to be retrieved by the view.

        This function is called by the logic to update the data obtained by the robot to a specific frame in GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame that will show the data
            data {dict} -- Data to be shown
        """
        try:
            with self.__data_loc:
                self.data[frame_id] = data
        except Exception as e:
            logger.info(e)

    def get_data(self, frame_id):
        """Function to collect data retrieved by the robot for an specific frame of the GUI

        This function is called by the view to get the last updated data to be shown in the GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame.

        Returns:
            data -- Depending on the caller frame could be image data, laser data, etc.
        """
        try:
            with self.__data_loc:
                data = self.data.get(frame_id, None)
        except Exception:
            pass

        return data

    def update_pose3d(self, data):
        """Update the pose3D data retrieved from the robot

        Arguments:
            data {pose3d} -- 3D position of the robot in the environment
        """
        try:
            with self.__pose_loc:
                self.pose3D_data = data
        except Exception:
            pass

    def get_pose3D(self):
        """Function to collect the pose3D data updated in `update_pose3d` function.

        This method is called from the view to collect the pose data and display it in GUI.

        Returns:
            pose3d -- 3D position of the robot in the environment
        """
        return self.pose3D_data

    # Simulation and dataset
    def reset_carla_simulation(self):
        logger.info("Restarting simulation")

    def pause_carla_simulation(self):
        logger.info("Pausing simulation")
        self.pilot.stop_event.set()

    def unpause_carla_simulation(self):
        logger.info("Resuming simulation")
        self.pilot.stop_event.clear()

    def record_rosbag(self, topics, dataset_name):
        """Start the recording process of the dataset using rosbags

        Arguments:
            topics {list} -- List of topics to be recorde
            dataset_name {str} -- Path of the resulting bag file
        """

        if not self.recording:
            logger.info("Recording bag at: {}".format(dataset_name))
            self.recording = True
            command = "rosbag record -O " + dataset_name + " " + " ".join(topics) + " __name:=behav_bag"
            command = shlex.split(command)
            with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
                self.rosbag_proc = subprocess.Popen(command, stdout=out, stderr=err)
        else:
            logger.info("Rosbag already recording")
            self.stop_record()

    def stop_record(self):
        """Stop the rosbag recording process."""
        if self.rosbag_proc and self.recording:
            logger.info("Stopping bag recording")
            self.recording = False
            command = "rosnode kill /behav_bag"
            command = shlex.split(command)
            with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
                subprocess.Popen(command, stdout=out, stderr=err)
        else:
            logger.info("No bag recording")

    def reload_brain(self, brain, model=None):
        """Helper function to reload the current brain from the GUI.

        Arguments:
            brain {srt} -- Brain to be reloadaed.
        """
        logger.info("Reloading brain... {}".format(brain))

        self.pause_pilot()
        self.pilot.reload_brain(brain, model)

    # Helper functions (connection with logic)

    def set_pilot(self, pilot):
        self.pilot = pilot

    def stop_pilot(self):
        self.pilot.kill_event.set()

    def pause_pilot(self):
        self.pilot.stop_event.set()

    def resume_pilot(self):
        self.start_time = datetime.now()
        self.pilot.start_time = datetime.now()
        self.pilot.stop_event.clear()

    def initialize_robot(self):
        self.pause_pilot()
        self.pilot.initialize_robot()


    def record_metrics(self, metrics_record_dir_path, world_counter=None, brain_counter=None, repetition_counter=None):
        logger.info("Recording metrics bag at: {}".format(metrics_record_dir_path))

        self.pilot.brain_iterations_real_time = []

        self.time_str = time.strftime("%Y%m%d-%H%M%S")       
        if world_counter is not None:
            current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world[world_counter])
        else:
            current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world)
        if brain_counter is not None:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path[brain_counter])
        else:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path)
        self.experiment_metrics = {
            'timestamp': self.time_str,
            'world_launch_file': current_world_tail,
            'brain_file': current_brain_tail,
            'robot_type': self.pilot.configuration.robot_type,
            'carla_map': self.carla_map.name,
            'ego_vehicle': self.ego_vehicle.type_id
        }
        if hasattr(self.pilot.configuration, 'experiment_model'):
            if brain_counter is not None:
                self.experiment_metrics['experiment_model'] = self.pilot.configuration.experiment_model[brain_counter]
            else:
                self.experiment_metrics['experiment_model'] = self.pilot.configuration.experiment_model
        if hasattr(self.pilot.configuration, 'experiment_name'):
            self.experiment_metrics['experiment_name'] = self.pilot.configuration.experiment_name
            self.experiment_metrics['experiment_description'] = self.pilot.configuration.experiment_description
            if hasattr(self.pilot.configuration, 'experiment_timeouts'):
                self.experiment_metrics['experiment_timeout'] = self.pilot.configuration.experiment_timeouts[world_counter]
            else:
                self.experiment_metrics['experiment_timeout'] = CIRCUITS_TIMEOUTS[os.path.basename(self.experiment_metrics['world'])] * 1.1
            self.experiment_metrics['experiment_repetition'] = repetition_counter

        self.metrics_record_dir_path = metrics_record_dir_path
        os.mkdir(self.metrics_record_dir_path + self.time_str)
        self.experiment_metrics_filename = self.metrics_record_dir_path + self.time_str + '/' + self.time_str + '.bag'
        topics = ['/carla/ego_vehicle/odometry', '/carla/ego_vehicle/collision', '/carla/ego_vehicle/lane_invasion', '/clock']
        command = "rosbag record -O " + self.experiment_metrics_filename + " " + " ".join(topics) + " __name:=behav_metrics_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            self.proc = subprocess.Popen(command, stdout=out, stderr=err)

    def stop_recording_metrics(self):
        logger.info("Stopping metrics bag recording")
        end_time = time.time()

        mean_brain_iterations_real_time = sum(self.pilot.brain_iterations_real_time) / len(self.pilot.brain_iterations_real_time)
        brain_iterations_frequency_real_time = 1 / mean_brain_iterations_real_time
        target_brain_iterations_real_time = 1 / (self.pilot.time_cycle / 1000)
        suddenness_distance = sum(self.pilot.brains.active_brain.suddenness_distance) / len(self.pilot.brains.active_brain.suddenness_distance)

        if self.pilot.brains.active_brain.camera_0_first_image is not None:
            first_images = [self.pilot.brains.active_brain.camera_0_first_image,
                            self.pilot.brains.active_brain.camera_1_first_image,
                            self.pilot.brains.active_brain.camera_2_first_image,
                            self.pilot.brains.active_brain.camera_3_first_image,
                            self.pilot.brains.active_brain.camera_4_first_image
                            ]
            last_images = [self.pilot.brains.active_brain.camera_0_last_image,
                            self.pilot.brains.active_brain.camera_1_last_image,
                            self.pilot.brains.active_brain.camera_2_last_image,
                            self.pilot.brains.active_brain.camera_3_last_image,
                            self.pilot.brains.active_brain.camera_4_last_image
                            ]
        else:
            first_images = []
            last_images = []

        command = "rosnode kill /behav_metrics_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            subprocess.Popen(command, stdout=out, stderr=err)

        # Wait for rosbag file to be closed. Otherwise it causes error
        while os.path.isfile(self.experiment_metrics_filename + '.active'):
            pass

        self.experiment_metrics = metrics_carla.get_metrics(self.experiment_metrics, self.experiment_metrics_filename, self.map_waypoints, self.metrics_record_dir_path + self.time_str + '/' + self.time_str)

        if hasattr(self.pilot.brains.active_brain, 'inference_times'):
            self.pilot.brains.active_brain.inference_times = self.pilot.brains.active_brain.inference_times[10:-10]
            self.experiment_metrics['gpu_mean_inference_time'] = sum(self.pilot.brains.active_brain.inference_times) / len(self.pilot.brains.active_brain.inference_times)
            self.experiment_metrics['gpu_inference_frequency'] = 1 / self.experiment_metrics['gpu_mean_inference_time']
            self.experiment_metrics['gpu_inference'] = self.pilot.brains.active_brain.gpu_inference

        self.experiment_metrics['mean_brain_iterations_real_time'] = mean_brain_iterations_real_time
        self.experiment_metrics['brain_iterations_frequency_real_time'] = brain_iterations_frequency_real_time
        self.experiment_metrics['target_brain_iterations_real_time'] = target_brain_iterations_real_time
        self.experiment_metrics['suddenness_distance'] = suddenness_distance
        
        self.save_metrics(first_images, last_images)

        logger.info("* Experiment total real time -> " + str(end_time - self.pilot.pilot_start_time) + ' s')
        self.experiment_metrics['experiment_total_real_time'] = end_time - self.pilot.pilot_start_time

        logger.info("Stopping metrics bag recording")


    def save_metrics(self, first_images, last_images):        
        with open(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + '.json', 'w') as f:
            json.dump(self.experiment_metrics, f)
        logger.info("Metrics stored in JSON file")

        im = Image.fromarray(first_images[0])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_1.jpeg")
        im = Image.fromarray(first_images[1])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_2.jpeg")
        im = Image.fromarray(first_images[2])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_3.jpeg")
        im = Image.fromarray(first_images[3])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_4.jpeg")
        im = Image.fromarray(first_images[4])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_first_image_5.jpeg")

        im = Image.fromarray(first_images[0])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_1.jpeg")
        im = Image.fromarray(first_images[1])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_2.jpeg")
        im = Image.fromarray(first_images[2])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_3.jpeg")
        im = Image.fromarray(first_images[3])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_4.jpeg")
        im = Image.fromarray(first_images[4])
        im.save(self.metrics_record_dir_path + self.time_str + '/' + self.time_str + "_last_image_5.jpeg")


