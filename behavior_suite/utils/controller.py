import shlex
import subprocess
import threading

from utils.logger import logger

import rospy
from std_srvs.srv import Empty


class Controller:

    def __init__(self):
        pass
        self.data_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.data = {}
        self.pose3D_data = None
        self.recording = False

    # GUI update
    def update_frame(self, frame_id, data):
        try:
            with self.data_lock:
                self.data[frame_id] = data
        except Exception:
            pass

    def get_data(self, frame_id):
        try:
            with self.data_lock:
                data = self.data.get(frame_id, None)
                # self.data[frame_id] = None
        except Exception:
            pass

        return data

    def update_pose3d(self, data):
        try:
            with self.pose_lock:
                self.pose3D_data = data
        except Exception:
            pass

    def get_pose3D(self):
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
        logger.info("Reloading brain... {}".format(brain))
        self.pilot.reload_brain(brain)

    def set_pilot(self, pilot):
        self.pilot = pilot
