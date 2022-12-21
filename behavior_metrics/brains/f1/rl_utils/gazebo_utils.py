import random

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


def set_new_pose(circuit_positions_set):
    """
    Receives the set of circuit positions (specified in the yml configuration file)
    and returns the index of the selected random position.

    (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
    """
    position = random.choice(list(enumerate(circuit_positions_set)))[0]
    print(position)

    state = ModelState()
    state.model_name = "f1_renault"
    state.pose.position.x = circuit_positions_set[position][1]
    state.pose.position.y = circuit_positions_set[position][2]
    state.pose.position.z = circuit_positions_set[position][3]
    state.pose.orientation.x = circuit_positions_set[position][4]
    state.pose.orientation.y = circuit_positions_set[position][5]
    state.pose.orientation.z = circuit_positions_set[position][6]
    state.pose.orientation.w = circuit_positions_set[position][7]

    rospy.wait_for_service("/gazebo/set_model_state")
    try:
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        set_state(state)
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))
    return position
