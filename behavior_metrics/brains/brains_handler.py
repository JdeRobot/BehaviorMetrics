import importlib
import sys
import subprocess
import os

from abc import abstractmethod
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare, 
    GaussNoise, Lambda
)
from utils.noise import salt_and_pepper_noise, gaussian_noise



""" TODO: fix neural brains """


class Brains(object):

    def __init__(self, sensors, actuators, brain_path, controller, model=None, config=None):

        self.sensors = sensors
        self.actuators = actuators
        self.controller = controller
        self.brain_path = brain_path
        self.config = config
        if model:
            self.model = model
        try:
            if brain_path:
                self.load_brain(brain_path)
        except AttributeError as e:
            print('Invalid brain path: {}\n[ERROR] {}'.format(brain_path, e))
            exit(1)

    def load_brain(self, path, model=None):

        path_split = path.split("/")
        robot_type = path_split[-2]
        module_name = path_split[-1][:-3]  # removing .py extension
        import_name = 'brains.' + robot_type + '.' + module_name

        if robot_type == 'f1rl':
            from utils import environment
            environment.close_gazebo()
            exec(open(self.brain_path).read())
        else:
            if import_name in sys.modules:  # for reloading sake
                del sys.modules[import_name]
            module = importlib.import_module(import_name)
            Brain = getattr(module, 'Brain')
            if robot_type == 'drone':
                self.active_brain = Brain(handler=self, config=self.config)
            else:
                if model: 
                    self.active_brain = Brain(self.sensors, self.actuators, model=model, handler=self, config=self.config)
                elif hasattr(self, 'model'):
                    self.active_brain = Brain(self.sensors, self.actuators, model=self.model, handler=self, config=self.config)
                else: 
                    self.active_brain = Brain(self.sensors, self.actuators, handler=self, config=self.config)

    def get_image(self, camera_name):
        camera = self.sensors.get_camera(camera_name)
        return camera.getImage()

    def get_laser_data(self, laser_name):
        laser = self.sensors.get_laser(laser_name)
        return laser.getLaserData()

    def get_motors(self, motors_name):
        return self.actuators.get_motor(motors_name)

    def update_pose3d(self, pose_data):
        self.controller.update_pose3d(pose_data)

    def update_frame(self, frame_id, data):
        self.controller.update_frame(frame_id, data)
        # try:
        #     frame = self.viewer.main_view.get_frame(frame_id)
        #     frame.set_data(data)
        #     return frame
        # except AttributeError as e:
        #     print('Not found ', frame_id, 'ERROR: ', e)
        #     pass

    def transform_image(self, image, option):
        augmentation_option = Compose([])
        if option == 'rain':
            augmentation_option = Compose([
                RandomRain(slant_lower=-10, slant_upper=10,
                           drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                           blur_value=7, brightness_coefficient=0.7,
                           rain_type='torrential', always_apply=True)
            ])
        elif option == 'night':
            augmentation_option = Compose([
                RandomBrightness([-0.5, -0.5], always_apply=True)
            ])
        elif option == 'shadow':
            augmentation_option = Compose([RandomShadow(always_apply=True)])
        elif option == 'snow':
            augmentation_option = Compose([RandomSnow(always_apply=True)])
        elif option == 'fog':
            augmentation_option = Compose([RandomFog(always_apply=True)])
        elif option == 'sunflare':
            augmentation_option = Compose([RandomSunFlare(always_apply=True)])
        elif option == 'gaussian':
            augmentation_option = Compose([GaussNoise(var_limit = 1000, always_apply=True)])
            # augmentation_option = Compose([Lambda(name = 'gaussian_noise', image = gaussian_noise, always_apply = True)])
        elif option == 'salt&pepper':
            augmentation_option = Compose([Lambda(name = 'salt_pepper_noise', image = salt_and_pepper_noise, always_apply = True)])
        transformed_image = augmentation_option(image=image)
        transformed_image = transformed_image["image"]
        return transformed_image

    @abstractmethod
    def execute(self):
        pass