from brains.f1.rl_utils.env_type import EnvironmentType
from brains.f1.rl_utils.exceptions import NoValidEnvironmentType
from brains.f1.rl_utils.models.f1_env_camera import QlearnF1FollowLineEnvGazebo


class F1Env:
    def __new__(cls, **config):
        training_type = config.get("training_type")

        # Qlearn F1 FollowLine camera
        if training_type == EnvironmentType.qlearn_env_camera_follow_line.value:
            return QlearnF1FollowLineEnvGazebo(**config)
        else:
            raise NoValidEnvironmentType(training_type)
