#!/usr/bin/env python

""" This module contains the logic to make predictions of keras-based neural networks.

This is module contains a class called KerasPredictor which is responsible of making predictions based
on a specific model. This is a generic class used by all the keras-based models.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.

Original idea from: https://github.com/JdeRobot/DetectionSuite/blob/master/DeepLearningSuite/DeepLearningSuiteLib/python_modules/keras_detect.py
Code adapted by: fqez
Original code by: vmartinezf and chanfr
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras.models import load_model
from utils.logger import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class KerasPredictor:
    """This class takes care of the prediction of the models that controls the behaviors of the robots.

    It's used for keras-based models and it's meant to predict a value (tipically a velocity) based on image inputs.

    Attributes:
        sess {tf.Session} -- Tensorflow session
        graph {tf.Graph} -- Tensorflow graph
        model -- Keras model instance
        img_width {int} -- Input images width
        img_height {int} -- Input images height
    """

    def __init__(self, path_to_hdf5):
        """Constructor of the class.

        Arguments:
            path_to_hdf5 {str} -- Path to the model file.
        """

        # Obtain the graph
        logger.info("Loading keras model {}".format(path_to_hdf5))
        self.sess = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()
        tf.compat.v1.keras.backend.set_session(self.sess)
        self.model = tf.keras.models.load_model(path_to_hdf5)

        input_size = self.model.input.shape.as_list()
        self.img_height = input_size[1]
        self.img_width = input_size[2]

    def predict(self, img, type='classification'):
        """Make a prediction of a velocity based on an input images.

        The model takes the image one of the robot's cameras and makes a predictino based on that information.

        Arguments:
            img {robot.interfaces.camera.Image} -- Image obtanied from one of the robot's cameras.
            type {str} -- Specify if the network is a classification or regression network (default: 'classification')

        Returns:
            int, float -- Predicted class or value based on the image information depending on the type of network.
        """

        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        input_img = np.stack([img_resized], axis=0)

        with self.graph.as_default():
            set_session(self.sess)
            y_pred = self.model.predict(input_img)

        if type == 'classification':
            return [np.argmax(prediction) for prediction in y_pred][0]
        else:
            return [float(prediction[0]) for prediction in y_pred][0]
