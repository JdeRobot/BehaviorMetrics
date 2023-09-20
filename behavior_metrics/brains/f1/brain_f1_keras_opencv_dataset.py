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

        self.suddenness_distance = []
        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None

        if self.config['GPU'] is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf

        self.gpu_inference = True if tf.test.gpu_device_name() else False

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                print("File " + model + " cannot be found in " + PRETRAINED_MODELS)

            if self.config['UseOptimized']:
            
                if 'tflite' in model:
                    print("Using TF lite models.....")
                    self.net = tf.lite.Interpreter(model_path= PRETRAINED_MODELS + model)
                    self.net.allocate_tensors()
                    self.input_index = self.net.get_input_details()[0]["index"]
                    self.output_index = self.net.get_output_details()[0]["index"]
                    self.inf_func = self.optim_inference
                else:
                    print("Using TensorRT models.....")
                    self.net = tf.saved_model.load(PRETRAINED_MODELS + model)
                    self.infer = self.net.signatures['serving_default']
                    self.output_tensorname = list(self.infer.structured_outputs.keys())[0]
                    self.inf_func = self.tftrt_inference
                    
            else:
                self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
                print(self.net.summary())
                
        else:
            print("** Brain not loaded **")
            print("- Models path: " + PRETRAINED_MODELS)
            print("- Model: " + str(model))

    def update_frame(self, frame_id, data, current_angular_speed=None, previous_angular_speed=None, distance=None, inference_time=None):
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

        if inference_time:
            data = np.array(data, copy=True)
            cv2.putText(data, f'Inf time: {inference_time}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        self.handler.update_frame(frame_id, data)

    def optim_inference(self, img):
        """ Utilize the optimized models in `.tflite` format for inference

        Arguments:
            img {ndarray} -- Image to make prediction on
        Return:
            output -- prediction from the model
        """
        # Pre-processing
        self.net.set_tensor(self.input_index, img)
        # Run inference.
        self.net.invoke()
        # Post-processing
        output = self.net.get_tensor(self.output_index)

        return output

    def tftrt_inference(self, img):
        """ Utilize the TensorRT optimized model for inference

        Arguments:
            img {ndarray} -- Image to make prediction on
        Return:
            output -- prediction from the model
        """
        # Pre-processing
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        # Run inference
        output = self.infer(img)[self.output_tensorname]

        return output.numpy()

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        self.cont += 1

        image = self.camera.getImage().data
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        base_image = image

        if self.cont == 1:
            self.first_image = image

        image = self.handler.transform_image(image, self.config['ImageTranform'])
        self.update_frame('frame_0', image)

        try:
            if self.config['ImageCropped']:
                image = image[240:480, 0:640]
            if 'ImageSize' in self.config:
                img = cv2.resize(image, (self.config['ImageSize'][0], self.config['ImageSize'][1]))
            else:
                img = image
            orig = img
            if self.config['ImageNormalized']:
                AUGMENTATIONS_TEST = Compose([
                    Normalize()
                ])
                image = AUGMENTATIONS_TEST(image=img)
                img = image["image"]

            img = np.expand_dims(img, axis=0)
            
            if self.config['UseOptimized']:
                start_time = time.time()
                prediction = self.inf_func(img)
                self.inference_times.append(time.time() - start_time)
            else:
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

            self.update_frame('frame_3', base_image, inference_time=self.inference_times[-1])

            current_w_normalized = prediction_w
            if self.previous_v != None:
                a = np.array((prediction[0][0], prediction[0][1]))
                b = np.array((self.previous_v, self.previous_w))
                distance = np.linalg.norm(a - b)
                self.suddenness_distance.append(distance)
            self.previous_v = prediction[0][0]
            self.previous_w = prediction[0][1]

            if self.previous_w_normalized != None:
                self.update_frame('frame_2', base_image, current_w_normalized, self.previous_w_normalized, str(round(distance, 4)))
            self.previous_w_normalized = current_w_normalized


            if not self.config['UseOptimized']: # not available for optimized models
                # GradCAM from image
                i = np.argmax(prediction[0])
                cam = GradCAM(self.net, i)
                heatmap = cam.compute_heatmap(img)
                heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
                (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
                self.update_frame('frame_1', output)

        except Exception as err:
            print(err)
