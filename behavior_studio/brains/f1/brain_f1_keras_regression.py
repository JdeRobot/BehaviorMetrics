"""
    Robot: F1
    Framework: keras
    Number of networks: 2
    Network type: Regression
    Predicionts:
        linear speed(v)
        angular speed(w)

    This brain uses regression networks based on Keras framework to predict the linear and angular velocity
    of the F1 car. For that task it uses two different regression convolutional neural networks, one for v
    and another one for w
"""
from behaviorlib.keraslib.keras_predict import KerasPredictor
import cv2
from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH

PRETRAINED_MODELS = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR
MODEL_PILOTNET_V = 'model_pilotnet_v.h5'
MODEL_PILOTNET_W = 'model_pilotnet_w.h5'

from os import path

class Brain:
    """Specific brain for the f1 robot. See header."""

    def __init__(self, sensors, actuators, handler=None):
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
        
        if not path.exists(PRETRAINED_MODELS + MODEL_PILOTNET_V):
            print("File "+MODEL_PILOTNET_V+" cannot be found in " + PRETRAINED_MODELS)
        if not path.exists(PRETRAINED_MODELS + MODEL_PILOTNET_W):
            print("File "+MODEL_PILOTNET_W+" cannot be found in " + PRETRAINED_MODELS)
        
        self.net_v = KerasPredictor(PRETRAINED_MODELS + MODEL_PILOTNET_V)
        self.net_w = KerasPredictor(PRETRAINED_MODELS + MODEL_PILOTNET_W)

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def execute(self):
        """Main loop of the brain. This will be called iteratively each TIME_CYCLE (see pilot.py)"""

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        image = self.camera.getImage().data
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        prediction_v = self.net_v.predict(img, type='regression')
        prediction_w = self.net_w.predict(img, type='regression')

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(prediction_v)
            self.motors.sendW(prediction_w)

        self.update_frame('frame_0', image)
