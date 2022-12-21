from brains.f1.rl_utils.env_type import EnvironmentType
from brains.f1.rl_utils.exceptions import NoValidEnvironmentType


class F1Env:
    def __new__(cls, **config):
        cls.circuit = None
        cls.vel_pub = None
        cls.unpause = None
        cls.pause = None
        cls.reset_proxy = None
        cls.action_space = None
        cls.reward_range = None
        cls.model_coordinates = None
        cls.position = None

        training_type = config.get("training_type")

        # Qlearn F1 FollowLine camera
        if training_type == EnvironmentType.qlearn_env_camera_follow_line.value:
            from brains.f1.rl_utils.models.f1_env_camera import (
                QlearnF1FollowLineEnvGazebo,
            )

            return QlearnF1FollowLineEnvGazebo(**config)
        else:
            raise NoValidEnvironmentType(training_type)
