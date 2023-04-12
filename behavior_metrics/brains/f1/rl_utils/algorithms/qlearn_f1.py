import os
import pickle
import random
import time

import numpy as np


class QLearnF1:
    def __init__(
        self
    ):
        pass
    def inference(self, state):
        return np.argmax(self.q_table[int(state)])

    def load_table(self, file):
        self.q_table = np.load(file)
