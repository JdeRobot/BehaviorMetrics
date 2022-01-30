"""

 PYTHONPATH=$PYTHONPATH:/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks python driver.py -c default.yml -g

    Robot: F1
    Framework: keras
    Number of networks: 1
    Network type: None
    Predicionts:
        linear speed(v)
        angular speed(w)

"""
import sys

import numpy as np
import cv2
import time
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data._utils.collate import default_collate

from os import path

from net_config.net_config import NetConfig
from models.visual_control import VisualControl
from visual_control_utils.check_point_loader import load_best_model
from visual_control_utils.logits_conversion import from_logit_to_estimation, from_one_hot_to_class
from visual_control_utils.visualization import add_arrow_prediction
from brains.base.brain_base import BrainBase
import time


class Brain(BrainBase):
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

        print("Model: {}".format(model))
        super().__init__(sensors, actuators, model, handler, config)

        self.motors = self.actuators.get_motor('motors_0')
        self.camera = self.sensors.get_camera('camera_0')
        self.cont = 0
        self.inference_times = []

        if not model:
            model = [
                '/media/frivas/External/phd/version_6', '/media/frivas/External/phd/version_7']
            model = "/media/frivas/External/phd/version_1"
            model = "/media/frivas/External/phd/version_10"
            model = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_5"
            model = "/media/frivas/External/phd/version_3"

        if model:
            if isinstance(model, str):
                checkpoint_path_input = load_best_model(model)

                check_points_info = {
                    "combined": checkpoint_path_input,
                    # "w": checkpoint_path_input_w,
                    # "v": checkpoint_path_input_v
                }
            else:
                print("Input model: {}".format(model))
                check_points_info = {
                    "w": load_best_model(model[0]),
                    "v": load_best_model(model[1])
                }

            print("-----")
            print("Len model: {}".format(len(model)))
            print(model)
        else:
            model_path = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_10"
            checkpoint_path_input = load_best_model(model_path)

            check_points_info = {
                "combined": checkpoint_path_input,
                # "w": checkpoint_path_input_w,
                # "v": checkpoint_path_input_v
            }

        # sys.exit(0)

        checkpoint_path_input_v = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_7/checkpoints/rc-classification-epoch=12-val_acc=1.00-val_loss=0.09.ckpt"
        checkpoint_path_input_v = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_9/checkpoints/rc-classification-epoch=33-val_acc=0.98-val_loss=0.05.ckpt"
        checkpoint_path_input_w = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_6/checkpoints/rc-classification-epoch=19-val_acc=1.00-val_loss=0.03.ckpt"
        checkpoint_path_input_w = "/home/frivas/devel/mio/github/2017-phd-francisco-rivas/deep_learning/python/networks/lightning_logs/version_8/checkpoints/rc-classification-epoch=34-val_acc=0.98-val_loss=0.06.ckpt"

        self.device = "cuda:0"
        # self.device = "cpu"

        if "cuda" in self.device:
            map_location = {'cuda:0': 'cpu'}
        else:
            map_location = None

        self.gpu_inferencing = "cuda" in self.device

        self.models = {}
        self.net_configs = {}
        self.eval_transforms = {}
        for controller in check_points_info:

            config_path = os.path.join(os.path.dirname(check_points_info[controller]), "..", "config.yaml")
            self.net_configs[controller] = NetConfig(config_path)

            self.models[controller] = VisualControl.load_from_checkpoint(checkpoint_path=check_points_info[controller], dataset_path="", lr=5e-2,
                                                            base_size=self.net_configs[controller].base_size,
                                                            batch_size=self.net_configs[controller].batch_size, net_config=self.net_configs[controller], map_location=map_location)

            self.models[controller].to(self.device)
            self.models[controller].eval()
            self.models[controller].freeze()

        # prints the learning_rate you used in this checkpoint

            self.eval_transforms[controller] = A.Compose([
                A.LongestMaxSize(self.net_configs[controller].base_size, always_apply=True),
                A.PadIfNeeded(self.net_configs[controller].base_size, self.net_configs[controller].base_size, always_apply=True,
                              border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(),
                ToTensorV2()
            ])

        print("model done")

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def calculate_v_w(self, raw_prediction, net_config):
        '''
            === V ===
            * slow -> 3
            * moderate -> 6
            * fast -> 10
            * very fast -> 13
            === W ===
            * radically left -> 1.7
            * moderate left -> 0.75
            * slightly left -> 0.25
            * slight -> 0
            * slightly right -> - 0.25
            * moderate right -> - 0.75
            * radically right -> - 1.7
        '''

        w = 0
        v = 0

        if net_config.head_type == NetConfig.CLASSIFICATION_TYPE:
            y_hat = from_logit_to_estimation(raw_prediction, net_config)
            prediction = from_one_hot_to_class(y_hat, net_config)
            motors_info = net_config.get_real_values_from_estimation(prediction)
        else:
            raw_prediction = raw_prediction.numpy()
            motors_info = {}
            for idx, controller in enumerate(net_config.regression_data["controllers"]):
                motors_info[controller] = raw_prediction[idx]

        # print(motors_info)

        if "v" in motors_info:
            v = motors_info["v"]
            if v < 2:
                v = 2
            self.motors.sendV(v)
        if "w" in motors_info:
            self.motors.sendW(motors_info["w"])
        return motors_info

    def execute_imp(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        start_time = time.time()

        self.cont += 1
        image = self.camera.getImage().data

        motors_info = {}
        try:
            start_time = time.time()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for controller in self.models:
                x = self.eval_transforms[controller](image=image)

                predictions = self.models[controller](default_collate([x["image"]]).to(self.device))[0].cpu()
                controller_output = self.calculate_v_w(predictions, self.net_configs[controller])
                motors_info.update(controller_output)

            # print(time.time() - start_time)
            self.inference_times.append(time.time() - start_time)
            image_labels = add_arrow_prediction(image, motors_info)
            self.update_frame('frame_0', image_labels)


        except Exception as exc:
            print("ERROR  -> {}".format(exc))


        if self.cont == 1:
            self.first_image = image

        # try:
        #     image = image[240:480, 0:640]
        #     # img = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        #     img = cv2.resize(image, (32, 32))
        #     img = np.expand_dims(img, axis=0)
        #     img = img.reshape(-1, 32, 32, 1)
        #     start_time = time.time()
        #     self.calculate_v_w(33)
        #     self.inference_times.append(time.time() - start_time)
        # except Exception as err:
        #     print(err)

        # self.update_frame('frame_0', image)

