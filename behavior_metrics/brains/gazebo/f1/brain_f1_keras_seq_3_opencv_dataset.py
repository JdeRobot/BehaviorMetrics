"""
    Robot: F1
    Framework: keras
    Number of networks: 1
    Network type: None
    Predicionts:
        linear speed(v)
        angular speed(w)

"""

import cv2
import math
import numpy as np
import os
import tensorflow as tf
import time
from albumentations import (
    Compose, Normalize
)
from os import path
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from utils.gradcam.gradcam import GradCAM

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'gazebo/tf_models/'


class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, sensors, actuators, model=None, handler=None, config=None):
        """Constructor of the class.

        Arguments:
            sensors {robot.sensors.Sensors} -- Sensors instance of the robot
            actuators {robot.actuators.Actuators} -- Actuators instance of the robot

        Keyword Arguments:
            handler {brains.brain_handler.Brains} -- Handler of the current brain. Communication with the controller
            (default: {None})
        """
        self.motors = actuators.get_motor('motors_0')
        self.camera = sensors.get_camera('camera_0')
        self.handler = handler
        self.cont = 0
        self.inference_times = []
        self.config = config

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.suddenness_distance = []
        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None

        self.third_image = []

        if self.config['GPU'] is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.gpu_inference = True if tf.test.gpu_device_name() else False

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
            print(self.net.summary())
        else:
            print("** Brain not loaded **")
            print("- Models path: " + PRETRAINED_MODELS)
            print("- Model: " + str(model))

    def update_frame(self, frame_id, data, current_angular_speed=None, previous_angular_speed=None, distance=None):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if current_angular_speed:
            data = np.array(data, copy=True)

            x1, y1 = int(data.shape[:2][1] / 2), data.shape[:2][0]  # ancho, alto
            length = 200
            angle = (90 + int(math.degrees(-current_angular_speed))) * 3.14 / 180.0
            x2 = int(x1 - length * math.cos(angle))
            y2 = int(y1 - length * math.sin(angle))

            line_thickness = 10
            cv2.line(data, (x1, y1), (x2, y2), (0, 0, 0), thickness=line_thickness)
            length = 150
            angle = (90 + int(math.degrees(-previous_angular_speed))) * 3.14 / 180.0
            x2 = int(x1 - length * math.cos(angle))
            y2 = int(y1 - length * math.sin(angle))

            cv2.line(data, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness)
            if float(distance) > 0.01:
                cv2.putText(data, distance, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(data, distance, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        self.handler.update_frame(frame_id, data)

    def check_center(self, position_x):
        if (len(position_x[0]) > 1):
            x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
            not_found = False
        else:
            # The center of the line is in position 326
            x_middle = 326
            not_found = True
        return x_middle, not_found

    def get_point(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right) / 2 + left
        return int(mid)

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        self.cont += 1
        '''
        if type(self.previous_image) == int:
            self.previous_image = self.camera.getImage().data
            self.previous_timestamp = timestamp
        if (timestamp - self.previous_timestamp  >= 0.085):
            #print(timestamp)
            self.previous_image = self.camera.getImage().data
            self.previous_timestamp = timestamp
        image = self.previous_image
        '''

        image = self.camera.getImage().data
        base_image = image
        if self.cont == 1:
            self.first_image = image
        image = self.handler.transform_image(image, self.config['ImageTranform'])
        try:
            if self.config['ImageCropped']:
                image = image[240:480, 0:640]
            if 'ImageSize' in self.config:
                img = cv2.resize(image, (self.config['ImageSize'][0], self.config['ImageSize'][1]))
            else:
                img = image
            self.update_frame('frame_0', img)
            if self.config['ImageNormalized']:
                AUGMENTATIONS_TEST = Compose([
                    Normalize()
                ])
                image = AUGMENTATIONS_TEST(image=img)
                img = image["image"]

            if self.cont == 1:
                self.first_image_stack = img
            elif self.cont == 2:
                self.second_image_stack = img
            elif self.cont == 3:
                self.third_image_stack = img
            elif self.cont == 4:
                self.fourth_image_stack = img
            elif self.cont == 5:
                self.fifth_image_stack = img
            elif self.cont == 6:
                self.sixth_image_stack = img
            elif self.cont == 7:
                self.seventh_image_stack = img
            elif self.cont == 8:
                self.eigth_image_stack = img
            elif self.cont == 9:
                self.nineth_image_stack = img
            elif self.cont > 9:
                self.tenth_image_stack = img
                images_buffer = [self.first_image_stack, self.fifth_image_stack, self.tenth_image_stack]
                images_buffer = np.array(images_buffer)
                img = np.expand_dims(images_buffer, axis=0)

                self.first_image_stack = self.second_image_stack
                self.second_image_stack = self.third_image_stack
                self.third_image_stack = self.fourth_image_stack
                self.fourth_image_stack = self.fifth_image_stack
                self.fifth_image_stack = self.sixth_image_stack
                self.sixth_image_stack = self.seventh_image_stack
                self.seventh_image_stack = self.eigth_image_stack
                self.eigth_image_stack = self.nineth_image_stack
                self.nineth_image_stack = self.tenth_image_stack

                start_time = time.time()
                prediction = self.net.predict(img)
                self.inference_times.append(time.time() - start_time)
                if self.config['PredictionsNormalized']:
                    prediction_v = prediction[0][0] * (24 - (6.5)) + (6.5)
                    prediction_w = prediction[0][1] * (7.1 - (-7.1)) + (-7.1)
                else:
                    prediction_v = prediction[0][0]
                    prediction_w = prediction[0][1]
                if prediction_w != '' and prediction_w != '':
                    self.motors.sendV(prediction_v)
                    self.motors.sendW(prediction_w)

                current_w_normalized = prediction_w
                if self.previous_v != None:
                    a = np.array((prediction[0][0], prediction[0][1]))
                    b = np.array((self.previous_v, self.previous_w))
                    distance = np.linalg.norm(a - b)
                    self.suddenness_distance.append(distance)
                self.previous_v = prediction[0][0]
                self.previous_w = prediction[0][1]

                if self.previous_w_normalized != None and distance:
                    self.update_frame('frame_1', base_image, current_w_normalized, self.previous_w_normalized, str(round(distance, 4)))
                self.previous_w_normalized = current_w_normalized


        except Exception as err:
            print(err)
