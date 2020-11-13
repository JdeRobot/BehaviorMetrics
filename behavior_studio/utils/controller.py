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

from utils.logger import logger

import os
import time
import rosbag
import json
from std_msgs.msg import String
import numpy as np
from bagpy import bagreader
import pandas as pd
import shutil


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
            
            
    def record_stats(self, perfect_lap_filename, stats_record_dir_path):
        logger.info("Recording stats bag at: {}".format(stats_record_dir_path))
        self.record_stats = True
        self.start_time = datetime.now()
        current_world_head, current_world_tail = os.path.split(self.pilot.configuration.current_world)
        current_brain_head, current_brain_tail = os.path.split(self.pilot.brains.brain_path)
        self.metrics = {}
        self.metrics['world'] = current_world_tail
        self.metrics['brain_path'] = current_brain_tail
        self.metrics['robot_type'] = self.pilot.configuration.robot_type
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
        
        self.read_perfect_lap_rosbag()
        self.statistics = self.lap_percentage_completed()
        
        shutil.rmtree(self.stats_filename.split('.bag')[0])
        shutil.rmtree(self.pilot.configuration.stats_perfect_lap.split('.bag')[0])
        
        
    def read_perfect_lap_rosbag(self):
        bag_reader = bagreader(self.perfect_lap_filename)
        csvfiles = []
        for topic in bag_reader.topics:
            data = bag_reader.message_by_topic(topic)
            csvfiles.append(data)

        data_file = 'full-lap/F1ROS-odom.csv'
        dataframe_pose = pd.read_csv(data_file)
        checkpoints = []
        for index, row in dataframe_pose.iterrows():
            checkpoints.append(row)

        start_point = checkpoints[0]
        for x, point in enumerate(checkpoints):
            if x is not 0 and point['header.stamp.secs'] - 10 > start_point['header.stamp.secs'] and self.is_finish_line(point, start_point) :
                lap_point = point

        self.circuit_diameter = self.circuit__distance_completed(checkpoints, lap_point)
        
        
    def is_finish_line(self, point, start_point):
        current_point = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        start_point = np.array([start_point['pose.pose.position.x'], start_point['pose.pose.position.y']])

        dist = (start_point - current_point) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist < 0.5:
            return True
        return False

    def circuit__distance_completed(self, checkpoints, lap_point):
        previous_point = []
        diameter = 0
        for i, point in enumerate(checkpoints):
            current_point = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
            if i is not 0:
                dist = (previous_point - current_point) ** 2
                dist = np.sum(dist, axis=0)
                dist = np.sqrt(dist)
                diameter += dist
            if point is lap_point:
                break
            previous_point = np.array([point['pose.pose.position.x'], point['pose.pose.position.y']])
        return diameter
    
    def lap_percentage_completed(self):
        statistics = {}
        bag_reader = bagreader(self.stats_filename)
        csvfiles = []
        for topic in bag_reader.topics:
            data = bag_reader.message_by_topic(topic)
            csvfiles.append(data)
        
        data_file = self.stats_filename.split('.bag')[0] + '/F1ROS-odom.csv'
        dataframe_pose = pd.read_csv(data_file)
        checkpoints = []
        for index, row in dataframe_pose.iterrows():
            checkpoints.append(row)
            
        start_point = checkpoints[0]
        end_point = checkpoints[len(checkpoints)-1]
        statistics['completed_distance'] = self.circuit__distance_completed(checkpoints, end_point)
        statistics['percentage_completed'] = (statistics['completed_distance'] / self.circuit_diameter) * 100      
        if statistics['percentage_completed'] > 100:
            start_point = checkpoints[0]
            for x, point in enumerate(checkpoints):
                if x is not 0 and point['header.stamp.secs'] - 10 > start_point['header.stamp.secs'] and self.finish_line(point, start_point) :
                    lap_point = point
            
            seconds_start = start_point['header.stamp.secs']
            seconds_end = lap_point['header.stamp.secs']
            statistics['lap_seconds'] = seconds_end - seconds_start
            statistics['circuit_diameter'] = self.circuit_completed_distance(checkpoints, lap_point)
            statistics['average_speed'] = self.circuit_completed_distance(checkpoints, lap_point)/statistics['lap_seconds']
        
        return statistics

    def reload_brain(self, brain):
        """Helper function to reload the current brain from the GUI.

        Arguments:
            brain {srt} -- Brain to be reloadaed.
        """
        logger.info("Reloading brain... {}".format(brain))
        
        self.pause_pilot()
        self.pilot.reload_brain(brain)

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
