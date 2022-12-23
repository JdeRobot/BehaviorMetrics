from brains.f1.rl_utils.algorithms.algorithms_type import AlgorithmsType
from brains.f1.rl_utils.algorithms.exceptions import NoValidAlgorithmType
from brains.f1.rl_utils.algorithms.qlearn import QLearn

class InferencerFactory:
    def __new__(cls, config):

        algorithm = config.algorithm
        inference_file_name = config.inference_file

        if algorithm == AlgorithmsType.QLEARN.value:
            actions_file_name = config.actions_file

            brain = QLearn(config)
            brain.load_model(inference_file_name, actions_file_name)

            return brain

