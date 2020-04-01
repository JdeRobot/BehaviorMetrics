import threading
import rospy

from std_srvs.srv import Empty
from std_msgs.msg import String


class Controller:

    def __init__(self):
        pass
        self.data_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.data = {}
        self.pose3D_data = None

    ### GUI update

    def update_frame(self, frame_id, data):
        with self.data_lock:
            self.data[frame_id] = data
            # print(self.data[frame_id])

    def get_data(self, frame_id):
        with self.data_lock:
            data = self.data[frame_id]
            # self.data[frame_id] = None

        return data

    def update_pose3d(self, data):
        with self.pose_lock:
            self.pose3D_data = data

    def get_pose3D(self):
        return self.pose3D_data

    ### Simulation and dataset

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

    def reload_brain(self, brain):
        self.pilot.reload_brain(brain)


    def set_pilot(self, pilot):
        self.pilot = pilot
