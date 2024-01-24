import numpy as np
import tensorflow as tf

from keras.models import Sequential, load_model

from .loaders import (
    LoadGlobalParams,
)

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class DDPGF1:
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
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        sampled_actions = tf.squeeze(self.model(tf_prev_state))
        sampled_actions = sampled_actions.numpy()
        # sampled_actions = np.argmax(sampled_actions)
        return np.squeeze(sampled_actions)
