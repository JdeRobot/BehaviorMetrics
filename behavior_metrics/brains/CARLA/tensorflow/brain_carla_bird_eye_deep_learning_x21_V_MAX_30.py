#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
import carla
from os import path
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from utils.logger import logger
from traceback import print_exc

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'carla_tf_models/'

from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.framework.errors_impl import UnimplementedError
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)



class Brain:

    def __init__(self, sensors, actuators, handler, model, config=None):
        self.camera_0 = sensors.get_camera('camera_0')
        self.camera_1 = sensors.get_camera('camera_1')
        self.camera_2 = sensors.get_camera('camera_2')
        self.camera_3 = sensors.get_camera('camera_3')
        
        self.cameras_first_images = []

        self.pose = sensors.get_pose3d('pose3d_0')

        self.bird_eye_view = sensors.get_bird_eye_view('bird_eye_view_0')

        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config
        self.inference_times = []
        self.gpu_inference = True if tf.test.gpu_device_name() else False

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)

        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) # seconds
        world = client.get_world()
        
        time.sleep(5)
        self.vehicle = world.get_actors().filter('vehicle.*')[0]

        if model:
            if not path.exists(PRETRAINED_MODELS + model):
                logger.info("File " + model + " cannot be found in " + PRETRAINED_MODELS)
            logger.info("** Load TF model **")
            self.net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
            logger.info("** Loaded TF model **")
        else:
            logger.info("** Brain not loaded **")
            logger.info("- Models path: " + PRETRAINED_MODELS)
            logger.info("- Model: " + str(model))

        self.image_1 = 0
        self.image_2 = 0
        self.image_3 = 0
        self.image_4 = 0
        self.image_5 = 0
        self.image_6 = 0
        self.image_7 = 0
        self.image_8 = 0
        self.image_9 = 0
        self.image_10 = 0

        self.image_11 = 0
        self.image_12 = 0
        self.image_13 = 0
        self.image_14 = 0
        self.image_15 = 0
        self.image_16 = 0
        self.image_17 = 0
        self.image_18 = 0
        self.image_19 = 0
        self.image_20 = 0

        self.image_21 = 0
        self.image_22 = 0
        self.image_23 = 0
        self.image_24 = 0
        self.image_25 = 0
        self.image_26 = 0
        self.image_27 = 0
        self.image_28 = 0
        self.image_29 = 0
        self.image_30 = 0

        self.image_31 = 0
        self.image_32 = 0
        self.image_33 = 0
        self.image_34 = 0
        self.image_35 = 0
        self.image_36 = 0
        self.image_37 = 0
        self.image_38 = 0
        self.image_39 = 0
        self.image_40 = 0

        self.image_41 = 0
        self.image_42 = 0
        self.image_43 = 0
        self.image_44 = 0
        self.image_45 = 0
        self.image_46 = 0
        self.image_47 = 0
        self.image_48 = 0
        self.image_49 = 0
        self.image_50 = 0

        self.image_51 = 0
        self.image_52 = 0
        self.image_53 = 0
        self.image_54 = 0
        self.image_55 = 0
        self.image_56 = 0
        self.image_57 = 0
        self.image_58 = 0
        self.image_59 = 0
        self.image_60 = 0

        self.image_61 = 0
        self.image_62 = 0
        self.image_63 = 0
        self.image_64 = 0
        self.image_65 = 0
        self.image_66 = 0
        self.image_67 = 0
        self.image_68 = 0
        self.image_69 = 0
        self.image_70 = 0

        self.image_71 = 0
        self.image_72 = 0
        self.image_73 = 0
        self.image_74 = 0
        self.image_75 = 0
        self.image_76 = 0
        self.image_77 = 0
        self.image_78 = 0
        self.image_79 = 0
        self.image_80 = 0

        self.image_81 = 0
        self.image_82 = 0
        self.image_83 = 0
        self.image_84 = 0
        self.image_85 = 0
        self.image_86 = 0
        self.image_87 = 0
        self.image_88 = 0
        self.image_89 = 0
        self.image_90 = 0

        self.image_91 = 0
        self.image_92 = 0
        self.image_93 = 0
        self.image_94 = 0
        self.image_95 = 0
        self.image_96 = 0
        self.image_97 = 0
        self.image_98 = 0
        self.image_99 = 0
        self.image_100 = 0

        self.bird_eye_view_images = 0
        self.bird_eye_view_unique_images = 0


    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if data.shape[0] != data.shape[1]:
            if data.shape[0] > data.shape[1]:
                difference = data.shape[0] - data.shape[1]
                extra_left, extra_right = int(difference/2), int(difference/2)
                extra_top, extra_bottom = 0, 0
            else:
                difference = data.shape[1] - data.shape[0]
                extra_left, extra_right = 0, 0
                extra_top, extra_bottom = int(difference/2), int(difference/2)
                

            data = np.pad(data, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0)), mode='constant', constant_values=0)
            
        self.handler.update_frame(frame_id, data)

    def update_pose(self, pose_data):
        self.handler.update_pose3d(pose_data)

    def execute(self):
        image = self.camera_0.getImage().data
        image_1 = self.camera_1.getImage().data
        image_2 = self.camera_2.getImage().data
        image_3 = self.camera_3.getImage().data

        bird_eye_view_1 = self.bird_eye_view.getImage(self.vehicle)
        bird_eye_view_1 = cv2.cvtColor(bird_eye_view_1, cv2.COLOR_BGR2RGB)

        if self.cameras_first_images == []:
            self.cameras_first_images.append(image)
            self.cameras_first_images.append(image_1)
            self.cameras_first_images.append(image_2)
            self.cameras_first_images.append(image_3)
            self.cameras_first_images.append(bird_eye_view_1)

        self.cameras_last_images = [
            image,
            image_1,
            image_2,
            image_3,
            bird_eye_view_1
        ]

        self.update_frame('frame_1', image_1)
        self.update_frame('frame_2', image_2)
        self.update_frame('frame_3', image_3)

        self.update_frame('frame_0', bird_eye_view_1)
        
        self.update_pose(self.pose.getPose3d())

        image_shape=(50, 150)
        img_base = cv2.resize(bird_eye_view_1, image_shape)

        AUGMENTATIONS_TEST = Compose([
            Normalize()
        ])
        image = AUGMENTATIONS_TEST(image=img_base)
        img = image["image"]

        if type(self.image_1) is int:
            self.image_1 = img
        elif type(self.image_2) is int:
            self.image_2 = img
        elif type(self.image_3) is int:
            self.image_3 = img
        elif type(self.image_4) is int:
            self.image_4 = img
        elif type(self.image_5) is int:
            self.image_5 = img
        elif type(self.image_6) is int:
            self.image_6 = img
        elif type(self.image_7) is int:
            self.image_7 = img
        elif type(self.image_8) is int:
            self.image_8 = img
        elif type(self.image_9) is int:
            self.image_9 = img
        elif type(self.image_10) is int:
            self.image_10 = img
        elif type(self.image_11) is int:
            self.image_11 = img
        elif type(self.image_12) is int:
            self.image_12 = img
        elif type(self.image_13) is int:
            self.image_13 = img
        elif type(self.image_14) is int:
            self.image_14 = img
        elif type(self.image_15) is int:
            self.image_15 = img
        elif type(self.image_16) is int:
            self.image_16 = img
        elif type(self.image_17) is int:
            self.image_17 = img
        elif type(self.image_18) is int:
            self.image_18 = img
        elif type(self.image_19) is int:
            self.image_19 = img
        elif type(self.image_20) is int:
            self.image_20 = img
        elif type(self.image_21) is int:
            self.image_21 = img
        elif type(self.image_22) is int:
            self.image_22 = img
        elif type(self.image_23) is int:
            self.image_23 = img
        elif type(self.image_24) is int:
            self.image_24 = img
        elif type(self.image_25) is int:
            self.image_25 = img
        elif type(self.image_26) is int:
            self.image_26 = img
        elif type(self.image_27) is int:
            self.image_27 = img
        elif type(self.image_28) is int:
            self.image_28 = img
        elif type(self.image_29) is int:
            self.image_29 = img
        elif type(self.image_30) is int:
            self.image_30 = img
        elif type(self.image_31) is int:
            self.image_31 = img
        elif type(self.image_32) is int:
            self.image_32 = img
        elif type(self.image_33) is int:
            self.image_33 = img
        elif type(self.image_34) is int:
            self.image_34 = img
        elif type(self.image_35) is int:
            self.image_35 = img
        elif type(self.image_36) is int:
            self.image_36 = img
        elif type(self.image_37) is int:
            self.image_37 = img
        elif type(self.image_38) is int:
            self.image_38 = img

        elif type(self.image_39) is int:
            self.image_39 = img
        elif type(self.image_40) is int:
            self.image_40 = img
        elif type(self.image_41) is int:
            self.image_41 = img
        elif type(self.image_42) is int:
            self.image_42 = img
        elif type(self.image_43) is int:
            self.image_43 = img
        elif type(self.image_44) is int:
            self.image_44 = img
        elif type(self.image_45) is int:
            self.image_45 = img
        elif type(self.image_46) is int:
            self.image_46 = img
        elif type(self.image_47) is int:
            self.image_47 = img
        elif type(self.image_48) is int:
            self.image_48 = img

        elif type(self.image_49) is int:
            self.image_49 = img
        elif type(self.image_50) is int:
            self.image_50 = img

        elif type(self.image_51) is int:
            self.image_51 = img
        elif type(self.image_52) is int:
            self.image_52 = img
        elif type(self.image_53) is int:
            self.image_53 = img
        elif type(self.image_54) is int:
            self.image_54 = img
        elif type(self.image_55) is int:
            self.image_55 = img
        elif type(self.image_56) is int:
            self.image_56 = img
        elif type(self.image_57) is int:
            self.image_57 = img
        elif type(self.image_58) is int:
            self.image_58 = img

        elif type(self.image_59) is int:
            self.image_59 = img
        elif type(self.image_60) is int:
            self.image_60 = img

        elif type(self.image_61) is int:
            self.image_61 = img
        elif type(self.image_62) is int:
            self.image_62 = img
        elif type(self.image_63) is int:
            self.image_63 = img
        elif type(self.image_64) is int:
            self.image_64 = img
        elif type(self.image_65) is int:
            self.image_65 = img
        elif type(self.image_66) is int:
            self.image_66 = img
        elif type(self.image_67) is int:
            self.image_67 = img
        elif type(self.image_68) is int:
            self.image_68 = img

        elif type(self.image_69) is int:
            self.image_69 = img
        elif type(self.image_70) is int:
            self.image_70 = img

        elif type(self.image_71) is int:
            self.image_71 = img
        elif type(self.image_72) is int:
            self.image_72 = img
        elif type(self.image_73) is int:
            self.image_73 = img
        elif type(self.image_74) is int:
            self.image_74 = img
        elif type(self.image_75) is int:
            self.image_75 = img
        elif type(self.image_76) is int:
            self.image_76 = img
        elif type(self.image_77) is int:
            self.image_77 = img
        elif type(self.image_78) is int:
            self.image_78 = img

        elif type(self.image_79) is int:
            self.image_79 = img
        elif type(self.image_80) is int:
            self.image_80 = img

        elif type(self.image_81) is int:
            self.image_81 = img
        elif type(self.image_82) is int:
            self.image_82 = img
        elif type(self.image_83) is int:
            self.image_83 = img
        
        elif type(self.image_84) is int:
            self.image_84 = img
        elif type(self.image_85) is int:
            self.image_85 = img
        elif type(self.image_86) is int:
            self.image_86 = img
        elif type(self.image_87) is int:
            self.image_87 = img
        elif type(self.image_88) is int:
            self.image_88 = img

        elif type(self.image_89) is int:
            self.image_89 = img
        elif type(self.image_90) is int:
            self.image_90 = img

        elif type(self.image_91) is int:
            self.image_91 = img
        elif type(self.image_92) is int:
            self.image_92 = img
        elif type(self.image_93) is int:
            self.image_93 = img

        elif type(self.image_94) is int:
            self.image_94 = img
        elif type(self.image_95) is int:
            self.image_95 = img
        elif type(self.image_96) is int:
            self.image_96 = img
        elif type(self.image_97) is int:
            self.image_97 = img
        elif type(self.image_98) is int:
            self.image_98 = img

        elif type(self.image_99) is int:
            self.image_99 = img
        elif type(self.image_100) is int:
            self.image_100 = img
        

        

        else:
            self.bird_eye_view_images += 1
            if (self.image_100==img).all() == False:
                self.bird_eye_view_unique_images += 1
            self.image_1 = self.image_2
            self.image_2 = self.image_3
            self.image_3 = self.image_4
            self.image_4 = self.image_5
            self.image_5 = self.image_6
            self.image_6 = self.image_7
            self.image_7 = self.image_8
            self.image_8 = self.image_9
            self.image_9 = self.image_10

            self.image_10 = self.image_11
            self.image_11 = self.image_12
            self.image_12 = self.image_13
            self.image_13 = self.image_14
            self.image_14 = self.image_15
            self.image_15 = self.image_16
            self.image_16 = self.image_17
            self.image_17 = self.image_18
            self.image_18 = self.image_19

            self.image_19 = self.image_20
            self.image_20 = self.image_21
            self.image_21 = self.image_22
            self.image_22 = self.image_23
            self.image_23 = self.image_24
            self.image_24 = self.image_25
            self.image_25 = self.image_26
            self.image_26 = self.image_27
            self.image_27 = self.image_28

            self.image_28 = self.image_29
            self.image_29 = self.image_30
            self.image_30 = self.image_31
            self.image_31 = self.image_32
            self.image_32 = self.image_33
            self.image_33 = self.image_34
            self.image_34 = self.image_35
            self.image_35 = self.image_36
            self.image_36 = self.image_37

            self.image_37 = self.image_38
            self.image_38 = self.image_39
            self.image_39 = self.image_40

            self.image_40 = self.image_41
            self.image_41 = self.image_42
            self.image_42 = self.image_43
            self.image_43 = self.image_44
            self.image_44 = self.image_45
            self.image_45 = self.image_46
            self.image_46 = self.image_47
            self.image_47 = self.image_48
            self.image_48 = self.image_49
            self.image_49 = self.image_50

            self.image_50 = self.image_51
            self.image_51 = self.image_52
            self.image_52 = self.image_53
            self.image_53 = self.image_54
            self.image_54 = self.image_55
            self.image_55 = self.image_56
            self.image_56 = self.image_57
            self.image_57 = self.image_58
            self.image_58 = self.image_59
            self.image_59 = self.image_60

            self.image_60 = self.image_61
            self.image_61 = self.image_62
            self.image_62 = self.image_63
            self.image_63 = self.image_64
            self.image_64 = self.image_65
            self.image_65 = self.image_66
            self.image_66 = self.image_67
            self.image_67 = self.image_68
            self.image_68 = self.image_69
            self.image_69 = self.image_70
             
            self.image_70 = self.image_71
            self.image_71 = self.image_72
            self.image_72 = self.image_73
            self.image_73 = self.image_74
            self.image_74 = self.image_75
            self.image_75 = self.image_76
            self.image_76 = self.image_77
            self.image_77 = self.image_78
            self.image_78 = self.image_79
            self.image_79 = self.image_80
            
            self.image_80 = self.image_81
            self.image_81 = self.image_82
            self.image_82 = self.image_83
            self.image_83 = self.image_84
            self.image_84 = self.image_85

            self.image_85 = self.image_86
            self.image_86 = self.image_87
            self.image_87 = self.image_88
            self.image_88 = self.image_89
            self.image_89 = self.image_90
            
            self.image_90 = self.image_91
            self.image_91 = self.image_92
            self.image_92 = self.image_93
            self.image_93 = self.image_94
            self.image_94 = self.image_95

            self.image_95 = self.image_96
            self.image_96 = self.image_97
            self.image_97 = self.image_98
            self.image_98 = self.image_99
            self.image_99 = self.image_90
            self.image_100 = img

            #img = [self.image_1, self.image_4, self.image_9]
            #img = [self.image_1, self.image_4, self.image_9, self.image_14, self.image_19, self.image_24, self.image_29, self.image_34, self.image_39]
            img = [self.image_1, self.image_5, self.image_10, self.image_15, self.image_20, self.image_25, self.image_30, self.image_35, self.image_40,
                   self.image_45, self.image_50, self.image_55, self.image_60, self.image_65, self.image_70, self.image_75, self.image_80, self.image_85,
                   self.image_90, self.image_95, self.image_100]
            img = np.expand_dims(img, axis=0)

            start_time = time.time()
            try:
                prediction = self.net.predict(img, verbose=0)
                self.inference_times.append(time.time() - start_time)
                throttle = prediction[0][0]
                steer = prediction[0][1] * (1 - (-1)) + (-1)
                break_command = prediction[0][2]
                speed = self.vehicle.get_velocity()
                vehicle_speed = 3.6 * math.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

                if vehicle_speed > 30:
                    self.motors.sendThrottle(0.0)
                    self.motors.sendSteer(steer)
                    self.motors.sendBrake(break_command)
                else:
                    if vehicle_speed < 5:
                        self.motors.sendThrottle(1.0)
                        self.motors.sendSteer(0.0)
                        self.motors.sendBrake(0)
                    else:
                        self.motors.sendThrottle(0.75)
                        self.motors.sendSteer(steer)
                        self.motors.sendBrake(break_command)

            except NotFoundError as ex:
                logger.info('Error inside brain: NotFoundError!')
                logger.warning(type(ex).__name__)
                print_exc()
                raise Exception(ex)
            except UnimplementedError as ex:
                logger.info('Error inside brain: UnimplementedError!')
                logger.warning(type(ex).__name__)
                print_exc()
                raise Exception(ex)
            except Exception as ex:
                logger.info('Error inside brain: Exception!')
                logger.warning(type(ex).__name__)
                print_exc()
                raise Exception(ex)
            