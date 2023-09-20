from brains.f1.rl_utils.algorithms.algorithms_type import AlgorithmsType
from brains.f1.rl_utils.algorithms.qlearn_f1 import QLearnF1
from brains.f1.rl_utils.algorithms.dqn_f1 import DQNF1

class InferencerFactory:
    def __new__(cls, config):

        algorithm = config.algorithm
        inference_file_name = config.inference_file

        if algorithm == AlgorithmsType.QLEARN.value:

            brain = QLearnF1()
            brain.load_table(inference_file_name)

            return brain

        if algorithm == AlgorithmsType.DQN.value:
            brain = DQNF1(config.env)
            brain.load_inference_model(inference_file_name)

            return brain


