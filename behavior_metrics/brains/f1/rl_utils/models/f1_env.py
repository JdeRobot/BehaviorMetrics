import numpy as np
import rospy
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from brains.f1.rl_utils import gazebo_envs


class F1Env(gazebo_envs.GazeboEnv):
    def __init__(self, **config):
        gazebo_envs.GazeboEnv.__init__(self, config)
        self.circuit_name = config.get("circuit_name")
        self.circuit_positions_set = config.get("circuit_positions_set")
        self.alternate_pose = config.get("alternate_pose")

        self.vel_pub = rospy.Publisher("/F1ROS/cmd_vel", Twist, queue_size=5)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
        self.reward_range = (-np.inf, np.inf)
        self.model_coordinates = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self.position = None

        self._seed()

        self.start_pose = np.array(config.get("gazebo_start_pose"))
        self.start_random_pose = config.get("gazebo_random_start_pose")
        self.model_state_name = config.get("model_state_name")

    def render(self, mode="human"):
        pass

    def step(self, action):

        raise NotImplementedError

    def reset(self):

        raise NotImplementedError

    def inference(self, action):

        raise NotImplementedError
