from pydantic import BaseModel

from brains.gazebo.f1.rl_utils.algorithms import InferencerFactory



# TODO future iteration -> make it language agnostic. Right now it is imported and instantiated like a library.
# In the future, it could be launched, binded to a port or a topic, and just answering to what it is listening
class InferencerWrapper:
    def __init__(self, algorithm, inference_file, actions_file="", env=None):

        inference_params = {
            "algorithm": algorithm,
            "inference_file": inference_file,
            "actions_file": actions_file,
            "env": env,
        }

        # TODO make mandatory env to retrieve from there the actions (instead of saving/loading actions file)
        params = self.InferenceValidator(**inference_params)

        self.inferencer = InferencerFactory(params)

    class InferenceValidator(BaseModel):
        inference_file: str
        algorithm: str
        actions_file: str
        env: object

    def inference(self, state):
        return self.inferencer.inference(state)