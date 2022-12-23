from enum import Enum


class AlgorithmsType(Enum):
    QLEARN = "qlearn"
    QLEARN_MULTIPLE_STATES = "qlearn_multiple_states"
    DQN = "dqn"
    DDPG = "ddpg"
    MANUAL = "manual"
