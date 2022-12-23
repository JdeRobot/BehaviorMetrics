import gym
import rospy


class GazeboEnv(gym.Env):

    def __init__(self, config):
        self.robot_name = config.get("robot_name")
    
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
 
    def _seed(self):
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass
    