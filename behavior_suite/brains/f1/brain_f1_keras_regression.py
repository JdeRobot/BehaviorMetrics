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
from utils.constants import PRETRAINED_MODELS_DIR

class Brain:

    def __init__(self, sensors, actuators, handler=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera = sensors.get_camera('camera_0')
        self.handler = handler
        self.cont = 0
        self.net_v = KerasPredictor(PRETRAINED_MODELS_DIR + 'model_pilotnet_v.h5')
        self.net_w = KerasPredictor(PRETRAINED_MODELS_DIR + 'model_pilotnet_w.h5')

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def execute(self):

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        image = self.camera.getImage().data
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        prediction_v = self.net_v.predict(img)
        prediction_w = self.net_w.predict(img)
        print(prediction_v, prediction_w)

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(prediction_v)
            self.motors.sendW(prediction_w)

        self.update_frame('frame_0', image)
