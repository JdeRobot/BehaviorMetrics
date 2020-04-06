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


class Brain:

    def __init__(self, sensors, actuators, handler=None):
        self.cont = 0
        self.net_v = KerasPredictor('path_to_v')
        self.net_w = KerasPredictor('path_to_w')
        self.motors = self.get_motors('motors_0')
        self.handler = handler

    def load_brain(self, path):
        raise AttributeError("Brain object has no attribute 'load_brain'")

    def execute(self):
        image = self.get_image('camera_0')

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        prediction_v = self.net_v.predict(image)
        prediction_w = self.net_w.predict(image)

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(prediction_v)
            self.motors.sendW(prediction_w)
