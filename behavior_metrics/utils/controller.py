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
from datetime import datetime

import rospy
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from utils.logger import logger

import os
import time
import rosbag
import json
from std_msgs.msg import String

from utils import metrics

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class Controller:
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
                # self.data[frame_id] = None
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

    def reset_gazebo_simulation(self):
        logger.info("Restarting simulation")
        reset_physics = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_physics()

    def pause_gazebo_simulation(self):
        logger.info("Pausing simulation")
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        try:
            pause_physics()
        except Exception as ex:
            print(ex)
        self.pilot.stop_event.set()

    def unpause_gazebo_simulation(self):
        logger.info("Resuming simulation")
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        try:
            unpause_physics()
        except Exception as ex:
            print(ex)
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
            
            
    def record_stats(self, perfect_lap_filename, stats_record_dir_path, world_counter=None, brain_counter=None, repetition_counter=None):
        logger.info("Recording stats bag at: {}".format(stats_record_dir_path))
        self.start_time = datetime.now()
        current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world)
        if brain_counter is not None:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path[brain_counter])
        else:
            current_brain_head, current_brain_tail = os.path.split(self.pilot.configuration.brain_path)
        self.metrics = {}
        self.metrics['world'] = current_world_tail
        self.metrics['brain_path'] = current_brain_tail
        self.metrics['robot_type'] = self.pilot.configuration.robot_type
        if hasattr(self.pilot.configuration, 'experiment_model'):
            if brain_counter is not None:
                self.metrics['experiment_model'] = self.pilot.configuration.experiment_model[brain_counter]
            else:
                self.metrics['experiment_model'] = self.pilot.configuration.experiment_model
        if hasattr(self.pilot.configuration, 'experiment_name'):
            self.metrics['experiment_name'] = self.pilot.configuration.experiment_name
            self.metrics['experiment_description'] = self.pilot.configuration.experiment_description
            self.metrics['experiment_timeout'] = self.pilot.configuration.experiment_timeouts[world_counter]
            self.metrics['experiment_repetition'] = repetition_counter
            
        self.perfect_lap_filename = perfect_lap_filename
        self.stats_record_dir_path = stats_record_dir_path
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.stats_filename = timestr + '.bag'
        
        topic = '/F1ROS/odom'
        command = "rosbag record -O " + self.stats_filename + " " + topic + " __name:=behav_stats_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            self.proc = subprocess.Popen(command, stdout=out, stderr=err)
        
    def stop_record_stats(self):
        logger.info("Stopping stats bag recording")
        command = "rosnode kill /behav_stats_bag"
        command = shlex.split(command)
        with open("logs/.roslaunch_stdout.log", "w") as out, open("logs/.roslaunch_stderr.log", "w") as err:
            subprocess.Popen(command, stdout=out, stderr=err)

        # Wait for rosbag file to be closed. Otherwise it causes error
        while(os.path.isfile(self.stats_filename + '.active')):
            pass

        checkpoints = []
        metrics_str = json.dumps(self.metrics)
        with rosbag.Bag(self.stats_filename, 'a') as bag:
            metadata_msg = String(data=metrics_str)
            bag.write('/metadata', metadata_msg, rospy.Time(bag.get_end_time()))
        bag.close()        
        perfect_lap_checkpoints, circuit_diameter = metrics.read_perfect_lap_rosbag(self.perfect_lap_filename)
        self.lap_statistics = metrics.lap_percentage_completed(self.stats_filename, perfect_lap_checkpoints, circuit_diameter)
        logger.info("END ---- > Stopping stats bag recording")
        
    def save_time_stats(self, mean_iteration_time, mean_inference_time, frame_rate, gpu_inferencing, first_image):
        time_stats = {'mean_iteration_time': mean_iteration_time, 
                    'mean_inference_time': mean_inference_time, 
                    'frame_rate': frame_rate, 
                    'gpu_inferencing': gpu_inferencing}
        metrics_str = json.dumps(time_stats)
        stats_str = json.dumps(self.lap_statistics)
        with rosbag.Bag(self.stats_filename, 'a') as bag:
            metadata_msg = String(data=metrics_str)
            lap_stats_msg = String(data=stats_str)
            bag.write('/time_stats', metadata_msg, rospy.Time(bag.get_end_time()))
            bag.write('/lap_stats', lap_stats_msg, rospy.Time(bag.get_end_time()))
            if first_image is not None and first_image.shape == (480, 640, 3):
                rospy.loginfo('Image received and sent to /first_image')
                bag.write('/first_image', self.cvbridge.cv2_to_imgmsg(first_image), rospy.Time(bag.get_end_time()))
            else:
                rospy.loginfo('Error: Image Broken and /first_image Skipped: {}'.format(first_image))
        bag.close()
        

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
