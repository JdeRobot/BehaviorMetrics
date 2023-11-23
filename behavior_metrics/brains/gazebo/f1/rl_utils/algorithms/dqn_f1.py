
import numpy as np
import tensorflow as tf

from keras.models import load_model
from .loaders import (
    LoadGlobalParams,
)

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class DQNF1:
    def __init__(self, config=None):

        self.global_params = LoadGlobalParams(config)
        self.state_space = self.global_params.states

        pass

    def load_inference_model(self, models_dir):
        """ """
        path_inference_model = models_dir
        inference_model = load_model(path_inference_model, compile=False)
        self.model = inference_model

        return self

    def inference(self, state):
        if self.state_space == "image":
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[
                0
            ]
        else:
            return self.model(np.array([state]))[0]
