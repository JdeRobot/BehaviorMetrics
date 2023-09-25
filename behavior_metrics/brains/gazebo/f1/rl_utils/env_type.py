from enum import Enum


class EnvironmentType(Enum):
    qlearn_env_camera_follow_line = "qlearn_camera_follow_line"
    qlearn_env_camera_follow_lane = "qlearn_camera_follow_lane"
    qlearn_env_laser_follow_line = "qlearn_laser_follow_line"
    dqn_env_follow_line = "dqn_follow_line"
    dqn_env_follow_lane = "dqn_follow_lane"
    manual_env = "manual"
    ddpg_env_follow_line = "ddpg_follow_line"
    ddpg_env_follow_lane = "ddpg_follow_lane"
