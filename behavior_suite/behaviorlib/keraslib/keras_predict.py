import numpy as np
import sys
import time
import cv2

from keras.models import load_model
from keras.backend import set_session
import tensorflow as tf


class KerasPredictor:

    def __init__(self, path_to_hdf5):

        # Obtain the graph
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)

        self.model = load_model(path_to_hdf5)

        input_size = self.model.input.shape.as_list()
        self.img_height = input_size[1]
        self.img_width = input_size[2]
        print(self.img_width, self.img_height)

    def predict(self, img):

        print("Starting inference")

        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        input_img = np.stack([img_resized], axis=0)

        start_time = time.time()
        with self.graph.as_default():
            set_session(self.sess)
            y_pred = self.model.predict(input_img)

        pred = [np.argmax(prediction) for prediction in y_pred][0]

        print("Inference Time: " + str(time.time() - start_time) + " seconds")

        return pred
