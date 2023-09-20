#############################################
# - Task: Follow Line
# - Algorithm: DQN
# - actions: discrete
# - State: Simplified perception and raw image
#
############################################

from geometry_msgs.msg import Twist
import numpy as np

from brains.gazebo.f1.rl_utils.models.f1_env import F1Env
from .settings import F1GazeboTFConfig


class FollowLineDQNF1GazeboTF(F1Env):
    def __init__(self, **config):

        ###### init F1env
        F1Env.__init__(self, **config)
        ###### init class variables
        F1GazeboTFConfig.__init__(self, **config)

    def reset(self):
        from .reset import (
            Reset,
        )

        if self.state_space == "image":
            return Reset.reset_f1_state_image(self)
        else:
            return Reset.reset_f1_state_sp(self)

    def step(self, action, step):
        from .step import (
            StepFollowLine,
        )

        if self.state_space == "image":
            return StepFollowLine.step_followline_state_image_actions_discretes(
                self, action, step
            )
        else:
            return StepFollowLine.step_followline_state_sp_actions_discretes(
                self, action, step
            )