import os

import cv2
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model

from utils.logger import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KerasPredictor:

    def __init__(self, path_to_hdf5):

        # Obtain the graph
        logger.info("Loading keras model {}".format(path_to_hdf5))
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)

        self.model = load_model(path_to_hdf5)

        input_size = self.model.input.shape.as_list()
        self.img_height = input_size[1]
        self.img_width = input_size[2]

    def predict(self, img):

        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        input_img = np.stack([img_resized], axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            y_pred = self.model.predict(input_img)

        pred = [np.argmax(prediction) for prediction in y_pred][0]

        return pred
