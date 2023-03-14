import gym
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


class GazeboEnv(gym.Env):

    def __init__(self, config):
        self.robot_name = config.get("robot_name")
    
    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_simulation")
        try:
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
 
    def _seed(self):
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass


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

        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number
