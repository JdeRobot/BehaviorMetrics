import time
import threading
import rospy
import rosbag
import subprocess
import shlex
from datetime import datetime

from std_srvs.srv import Empty
from std_msgs.msg import String

from robot.sensors import Sensors
from robot.actuators import Actuators
from brains.brains_handler import Brains
import conf.environment as env

TIME_CYCLE = 80

class RosSrvHandler:

    def __init__(self):
        self.recording = False
        self.rosbag_proc = None

    def pause_gazebo_simulation(self):
        print("Pausing gazebo simulation...")
        pause_physics = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        pause_physics()

    def unpause_gazebo_simulation(self):
        print("UNPausing gazebo simulation...")
        unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        unpause_physics()

    def record_rosbag(self, topics, dataset_name):
        if not self.recording:
            self.recording = True
            dataset_name = 'testbag'
            topics = ['/F1ROS/cmd_vel', '/F1ROS/cameraL/image_raw']
            command = "rosbag record -O datasets/" + dataset_name + " " +" ".join(topics)
            command = shlex.split(command)
            self.rosbag_proc = subprocess.Popen(command)

        else:
            print("Rosbag record already running")

    def stop_record(self):
        if self.rosbag_proc and self.recording:
            print('Stopping rosbag record')
            self.recording = False
            self.rosbag_proc.terminate()
        else:
            print("No bags recording")


class Pilot(threading.Thread):

    def __init__(self, config_data):
        robot = config_data['Behaviors']['Robot']
        sensors_config = robot['Sensors']
        actuators_config = robot['Actuators']
        brain_path = robot['BrainPath']

        self.sensors = Sensors(sensors_config)
        self.actuators = Actuators(actuators_config)
        self.brains = Brains(self.sensors, self.actuators, brain_path)

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        threading.Thread.__init__(self, args=self.stop_event)
        
        thread_ui_comm = threading.Thread(target=self.ui_listener)
        thread_ui_comm.daemon = True
        thread_ui_comm.start()

        self.ros_handler = RosSrvHandler()


    def run(self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.brains.active_brain.execute()
            dt = datetime.now() - start_time

            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if (ms < TIME_CYCLE):
                time.sleep((TIME_CYCLE - ms) / 1000.0)

    def stop(self):
        self.stop_event.set()

    def play(self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        self.actuators.kill()
        self.kill_event.set()
    
    def callback(self,data):
        if data.data == env.PAUSE_SIMULATION:
            self.ros_handler.pause_gazebo_simulation()
        elif data.data == env.RESUME_SIMULATION:
            self.ros_handler.unpause_gazebo_simulation()
        elif data.data == env.CHANGE_BRAIN:
            self.ros_handler.pause_gazebo_simulation()
            self.brains.load_brain('brains/f1/brain_f1_opencv2.py')
            self.ros_handler.unpause_gazebo_simulation()
        elif data.data == env.RECORD_DATASET:
            self.ros_handler.record_rosbag(None, None)
        elif data.data == env.STOP_RECORD_DATASET:
            self.ros_handler.stop_record()
        elif data.data == 'quit':
            self.kill()
            
    def ui_listener(self):
        rospy.Subscriber("/behavior/ui_comm", String, self.callback)
        rospy.spin()