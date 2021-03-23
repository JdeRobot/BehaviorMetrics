import importlib
import sys
from abc import abstractmethod
import subprocess
import os


""" TODO: fix neural brains """


class Brains(object):

    def __init__(self, sensors, actuatrors, brain_path, controller, model=None):

        self.sensors = sensors
        self.actuatrors = actuatrors
        self.controller = controller
        self.brain_path = brain_path
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
            if model: 
                self.active_brain = Brain(self.sensors, self.actuatrors, model=model, handler=self)
            elif hasattr(self, 'model'):
                self.active_brain = Brain(self.sensors, self.actuatrors, model=self.model, handler=self)
            else: 
                self.active_brain = Brain(self.sensors, self.actuatrors, handler=self)

    def get_image(self, camera_name):
        camera = self.sensors.get_camera(camera_name)
        return camera.getImage()

    def get_laser_data(self, laser_name):
        laser = self.sensors.get_laser(laser_name)
        return laser.getLaserData()

    def get_motors(self, motors_name):
        return self.actuatrors.get_motor(motors_name)

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

    @abstractmethod
    def execute(self):
        pass
