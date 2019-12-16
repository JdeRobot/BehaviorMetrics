#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model


class ClassificationNetwork():
    def __init__(self, net_model):
        # Load model
        self.model_file_v = 'Net/Keras/' + net_model['Model_Classification_v']
        self.model_file_w = 'Net/Keras/' + net_model['Model_Classification_w']
        self.model_v = load_model(self.model_file_v)
        self.model_w = load_model(self.model_file_w)

        # Obtain the graph
        self.graph = tf.get_default_graph()

        # The Keras network works on 160x120
        #self.img_height = 120
        # If we have cropped images, the network works on 160x60
        self.img_height = 60
        self.img_width = 160

        self.num_classes_w = 7

        self.prediction_v = ""
        self.prediction_w = ""


    def setCamera(self, camera):
        self.camera = camera


    def convertLabel_w(self, label):
        if self.num_classes_w == 2:
            if label == 0:
                string_label = "left"
            else:
                string_label = "right"
        elif self.num_classes_w == 7:
            if label == 0:
                string_label = "radically_left"
            elif label == 1:
                string_label = "moderately_left"
            elif label == 2:
                string_label = "slightly_left"
            elif label == 3:
                string_label = "slight"
            elif label == 4:
                string_label = "slightly_right"
            elif label == 5:
                string_label = "moderately_right"
            elif label == 6:
                string_label = "radically_right"
        elif self.num_classes_w == 9:
            if label == 0:
                string_label = "radically_left"
            elif label == 1:
                string_label = "strongly_left"
            elif label == 2:
                string_label = "moderately_left"
            elif label == 3:
                string_label = "slightly_left"
            elif label == 4:
                string_label = "slight"
            elif label == 5:
                string_label = "slightly_right"
            elif label == 6:
                string_label = "moderately_right"
            elif label == 7:
                string_label = "strongly_right"
            elif label == 8:
                string_label = "radically_right"
        return string_label


    def convertLabel_v(self, label):
        if label == 0:
            string_label = "slow"
        elif label == 1:
            string_label = "moderate"
        elif label == 2:
            string_label = "fast"
        elif label == 3:
            string_label = "very_fast"
        elif label == 4:
            string_label = "negative"
        return string_label


    def predict(self):
        input_image = self.camera.getImage()

        # Preprocessing
        #img = cv2.cvtColor(input_image.data, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(input_image.data[240:480, 0:640], cv2.COLOR_RGB2BGR)
        if img is not None:
            img_resized = cv2.resize(img, (self.img_width, self.img_height))

            # We adapt the image
            input_img = np.stack([img_resized], axis=0)

            # While predicting, use the same graph
            with self.graph.as_default():
                # Make prediction
                predictions_v = self.model_v.predict(input_img)
                predictions_w = self.model_w.predict(input_img)
            y_pred_v = [np.argmax(prediction) for prediction in predictions_v][0]
            y_pred_w = [np.argmax(prediction) for prediction in predictions_w][0]

            # Convert int prediction to corresponded label
            y_pred_v = self.convertLabel_v(y_pred_v)
            y_pred_w = self.convertLabel_w(y_pred_w)

            self.prediction_v = y_pred_v
            self.prediction_w = y_pred_w
