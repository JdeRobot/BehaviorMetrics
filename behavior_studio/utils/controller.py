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

import rospy
from std_srvs.srv import Empty

from utils.logger import logger

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

    def __init__(self, headless=False):
        """ Constructor of the class. """
        pass
        self.__data_loc = threading.Lock()
        self.__pose_loc = threading.Lock()
        self.data = {}
        self.pose3D_data = None
        self.recording = False
        self.headless = headless

    def create_connection(self):
        """ Create a python socket connection and bind it to the real robot to start sharing messages.

        This class will manage all the data trasfer between the GUI and the real robot if launched with
        headless mode."""
        pass

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
        pause_physics()
        self.pilot.stop_event.set()

    def unpause_gazebo_simulation(self):
        logger.info("Resuming simulation")
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_physics()
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
            topics = ['/F1ROS/cmd_vel', '/F1ROS/cameraL/image_raw']
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

    def reload_brain(self, brain):
        """Helper function to reload the current brain from the GUI.

        Arguments:
            brain {srt} -- Brain to be reloadaed.
        """
        logger.info("Reloading brain... {}".format(brain))
        self.pause_pilot()
        self.pilot.reload_brain(brain)
        self.resume_pilot()

    # Helper functions (connection with logic)

    def set_pilot(self, pilot):
        self.pilot = pilot

    def stop_pilot(self):
        self.pilot.kill_event.set()

    def pause_pilot(self):
        self.pilot.stop_event.set()

    def resume_pilot(self):
        self.pilot.stop_event.clear()

    def initialize_robot(self):
        self.pause_pilot()
        self.pilot.initialize_robot()
        self.resume_pilot()
