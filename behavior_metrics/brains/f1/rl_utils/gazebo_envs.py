from pathlib import Path
import os
import random
import signal
import subprocess
import sys

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import gym
import numpy as np
from rosgraph_msgs.msg import Clock
import rospy
from tf.transformations import quaternion_from_euler

#from agents.utils import print_messages
class Bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def print_messages(*args, **kwargs):

    print(f"\n\t{Bcolors.OKCYAN}====>\t{args[0]}:{Bcolors.ENDC}\n")
    for key, value in kwargs.items():
        print(f"\t{Bcolors.OKBLUE}[INFO] {key} = {value}{Bcolors.ENDC}")
    print("\n")



class GazeboEnv(gym.Env):
    """
    Superclass for all Gazebo environments.
    """

    metadata = {"render.models": ["human"]}

    def __init__(self, config):
        # print(config.get("launchfile"))
        self.last_clock_msg = Clock()
        self.port = "11311"  # str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = "11345"  # str(random_number+1) #os.environ["ROS_PORT_SIM"]

        self.robot_name = config.get("robot_name")

        # print(f"\nROS_MASTER_URI = http://localhost:{self.port}\n")
        # print(f"GAZEBO_MASTER_URI = http://localhost:{self.port_gazebo}\n")

        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")

        if config.get("launch_absolute_path") != None:
            fullpath = config.get("launch_absolute_path")
        else:
            # TODO: Global env for 'my_env'. It must be passed in constructor.
            fullpath = '/usr/local/gazebo/launch/simple_circuit.launch'
            '''
            fullpath = str(
                Path(
                    Path(__file__).resolve().parents[2]
                    / "CustomRobots"
                    / config.get("environment_folder")
                    / "launch"
                    / config.get("launchfile")
                )
            )
            '''
            # print(f"-----> {fullpath}")
        if not os.path.exists(fullpath):
            raise IOError(f"File {fullpath} does not exist")

        self._roslaunch = subprocess.Popen(
            [
                sys.executable,
                os.path.join(ros_path, b"roslaunch"),
                "-p",
                self.port,
                fullpath,
            ]
        )
        # print("Gazebo launched!")

        self.gzclient_pid = 0
        # Launch the simulation with the given launchfile name
        #rospy.init_node("gym", anonymous=True)

        ################################################################################################################
        # r = rospy.Rate(1)
        # self.clock_sub = rospy.Subscriber('/clock', Clock, self.callback, queue_size=1000000)
        # while not rospy.is_shutdown():
        #     print("initialization: ", rospy.rostime.is_rostime_initialized())
        #     print("Wallclock: ", rospy.rostime.is_wallclock())
        #     print("Time: ", time.time())
        #     print("Rospyclock: ", rospy.rostime.get_rostime().secs)
        #     # print("/clock: ", str(self.last_clock_msg))
        #     last_ros_time_ = self.last_clock_msg
        #     print("Clock:", last_ros_time_)
        #     # print("Waiting for synch with ROS clock")
        #     # if wallclock == False:
        #     #     break
        #     r.sleep()
        ################################################################################################################

    # def callback(self, message):
    #     """
    #     Callback method for the subscriber of the clock topic
    #     :param message:
    #     :return:
    #     """
    #     # self.last_clock_msg = int(str(message.clock.secs) + str(message.clock.nsecs)) / 1e6
    #     # print("Message", message)
    #     self.last_clock_msg = message
    #     # print("Message", message)

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def _gazebo_get_agent_position(self):

        object_coordinates = self.model_coordinates(self.robot_name, "")
        x_position = round(object_coordinates.pose.position.x, 2)
        y_position = round(object_coordinates.pose.position.y, 2)

        print_messages(
            "en _gazebo_get_agent_position()",
            robot_name=self.robot_name,
            object_coordinates=object_coordinates,
        )
        return x_position, y_position

    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
        except rospy.ServiceException as e:
            print(f"/gazebo/reset_simulation service call failed: {e}")

    def _gazebo_pause(self):
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print(f"/gazebo/pause_physics service call failed: {e}")

    def _gazebo_unpause(self):
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(f"/gazebo/unpause_physics service call failed: {e}")

    def _gazebo_set_new_pose(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
        self.position = pos

        pos_number = self.circuit["gaz_pos"][0]

        state = ModelState()
        state.model_name = self.config.get("robot_name")
        state.pose.position.x = self.circuit["gaz_pos"][pos][1]
        state.pose.position.y = self.circuit["gaz_pos"][pos][2]
        state.pose.position.z = self.circuit["gaz_pos"][pos][3]
        state.pose.orientation.x = self.circuit["gaz_pos"][pos][4]
        state.pose.orientation.y = self.circuit["gaz_pos"][pos][5]
        state.pose.orientation.z = self.circuit["gaz_pos"][pos][6]
        state.pose.orientation.w = self.circuit["gaz_pos"][pos][7]

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_set_new_pose_robot(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        # pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
        # self.position = pos

        pos_number = 0

        state = ModelState()
        state.model_name = self.robot_name
        state.pose.position.x = self.reset_pos_x
        state.pose.position.y = self.reset_pos_y
        state.pose.position.z = self.reset_pos_z
        state.pose.orientation.x = 0
        state.pose.orientation.y = 0
        state.pose.orientation.z = 0
        state.pose.orientation.w = 0

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_set_fix_pose_autoparking(self):
        """
        https://stackoverflow.com/questions/60840019/practical-understanding-of-quaternions-in-ros-moveit
        """
        pos_number = self.start_pose
        # pos_number = self.start_random_pose[posit][0]
        # pos_number = self.gazebo_random_start_pose[posit][0]

        state = ModelState()
        # state.model_name = "f1_renault"
        state.model_name = self.model_state_name

        # Pose Position
        state.pose.position.x = self.start_pose[0][0]
        state.pose.position.y = self.start_pose[0][1]
        state.pose.position.z = self.start_pose[0][2]

        # Pose orientation
        quaternion = quaternion_from_euler(
            self.start_pose[0][3], self.start_pose[0][4], self.start_pose[0][5]
        )

        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]

        print_messages(
            "en _gazebo_set_fix_pose_autoparking()",
            start_pose=self.start_pose,
            start_pose0=self.start_pose[0][0],
            start_pose1=self.start_pose[0][1],
            start_pose2=self.start_pose[0][2],
            start_pose3=self.start_pose[0][3],
            start_pose4=self.start_pose[0][4],
            start_pose5=self.start_pose[0][5],
            state_pose_orientation=state.pose.orientation,
            # start_pose6=self.start_pose[0][6],
            # circuit_positions_set=self.circuit_positions_set,
            start_random_pose=self.start_random_pose,
            # gazebo_random_start_pose=self.gazebo_random_start_pose,
            model_state_name=self.model_state_name,
        )

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_set_random_pose_autoparking(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        random_init = np.random.randint(0, high=len(self.start_random_pose))
        # pos_number = self.start_random_pose[posit][0]
        pos_number = self.start_random_pose[random_init][0]

        state = ModelState()
        state.model_name = self.model_state_name
        # Pose Position
        state.pose.position.x = self.start_random_pose[random_init][0]
        state.pose.position.y = self.start_random_pose[random_init][1]
        state.pose.position.z = self.start_random_pose[random_init][2]
        # Pose orientation
        quaternion = quaternion_from_euler(
            self.start_random_pose[random_init][3],
            self.start_random_pose[random_init][4],
            self.start_random_pose[random_init][5],
        )
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]

        print_messages(
            "en _gazebo_set_random_pose_autoparking()",
            random_init=random_init,
            start_random_pose=self.start_random_pose,
            start_pose=self.start_pose,
            start_random_pose0=self.start_random_pose[random_init][0],
            start_random_pose1=self.start_random_pose[random_init][1],
            start_random_pose2=self.start_random_pose[random_init][2],
            start_random_pose3=self.start_random_pose[random_init][3],
            start_random_pose4=self.start_random_pose[random_init][4],
            start_random_pose5=self.start_random_pose[random_init][5],
            state_pose_position=state.pose.position,
            state_pose_orientation=state.pose.orientation,
            model_state_name=self.model_state_name,
        )

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_set_fix_pose_f1_follow_right_lane(self):
        pos_number = self.start_pose
        state = ModelState()
        state.model_name = self.model_state_name
        # Pose Position
        state.pose.position.x = self.start_pose[0][0]
        state.pose.position.y = self.start_pose[0][1]
        state.pose.position.z = self.start_pose[0][2]

        # Pose orientation
        state.pose.orientation.x = self.start_pose[0][3]
        state.pose.orientation.y = self.start_pose[0][4]
        state.pose.orientation.z = self.start_pose[0][5]
        state.pose.orientation.w = self.start_pose[0][6]

        # print_messages(
        #    "en _gazebo_set_fix_pose_f1_follow_right_lane()",
        #    start_pose=self.start_pose,
        #    start_pose0=self.start_pose[0][0],
        #    start_pose1=self.start_pose[0][1],
        #    start_pose2=self.start_pose[0][2],
        #    start_pose3=self.start_pose[0][3],
        #    start_pose4=self.start_pose[0][4],
        #    start_pose5=self.start_pose[0][5],
        #    start_pose6=self.start_pose[0][6],
        #    state_pose_orientation=state.pose.orientation,
        #    # start_pose6=self.start_pose[0][6],
        #    # circuit_positions_set=self.circuit_positions_set,
        #    start_random_pose=self.start_random_pose,
        #    # gazebo_random_start_pose=self.gazebo_random_start_pose,
        #    model_state_name=self.model_state_name,
        # )

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_set_random_pose_f1_follow_rigth_lane(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        random_init = np.random.randint(0, high=len(self.start_random_pose))
        # pos_number = self.start_random_pose[posit][0]
        pos_number = self.start_random_pose[random_init][0]

        state = ModelState()
        state.model_name = self.model_state_name
        # Pose Position
        state.pose.position.x = self.start_random_pose[random_init][0]
        state.pose.position.y = self.start_random_pose[random_init][1]
        state.pose.position.z = self.start_random_pose[random_init][2]
        # Pose orientation
        state.pose.orientation.x = self.start_random_pose[random_init][3]
        state.pose.orientation.y = self.start_random_pose[random_init][4]
        state.pose.orientation.z = self.start_random_pose[random_init][5]
        state.pose.orientation.w = self.start_random_pose[random_init][6]

        # quaternion = quaternion_from_euler(
        #    self.start_random_pose[random_init][3],
        #    self.start_random_pose[random_init][4],
        #    self.start_random_pose[random_init][5],
        # )
        # state.pose.orientation.x = quaternion[0]
        # state.pose.orientation.y = quaternion[1]
        # state.pose.orientation.z = quaternion[2]
        # state.pose.orientation.w = quaternion[3]

        # print_messages(
        #    "en _gazebo_set_random_pose_f1_follow_rigth_lane()",
        #    random_init=random_init,
        #    start_random_pose=self.start_random_pose,
        #    start_pose=self.start_pose,
        #    start_random_pose0=self.start_random_pose[random_init][0],
        #    start_random_pose1=self.start_random_pose[random_init][1],
        #    start_random_pose2=self.start_random_pose[random_init][2],
        #    start_random_pose3=self.start_random_pose[random_init][3],
        #    start_random_pose4=self.start_random_pose[random_init][4],
        #    start_random_pose5=self.start_random_pose[random_init][5],
        #    state_pose_position=state.pose.position,
        #    state_pose_orientation=state.pose.orientation,
        #    model_state_name=self.model_state_name,
        # )

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _render(self, mode="human", close=False):

        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count("gzclient")
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count("gzclient")
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(
                subprocess.check_output(["pidof", "-s", "gzclient"])
            )
        else:
            self.gzclient_pid = 0

    @staticmethod
    def _close():

        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count("gzclient")
        gzserver_count = tmp.count("gzserver")
        roscore_count = tmp.count("roscore")
        rosmaster_count = tmp.count("rosmaster")

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0:
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass

    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
