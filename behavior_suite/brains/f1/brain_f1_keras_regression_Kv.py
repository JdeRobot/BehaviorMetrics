"""
    Robot: F1
    Framework: keras
    Number of networks: 1
    Network type: Regression
    Predicionts:
        angular speed(w)

    This brain uses a regression network based on Keras framework to predict the angular velocity of the F1 car.
    For that task it uses a regression convolutional neural network for w leaving the linear speed constant
"""
from brains import Brains
from behaviorlib.keras.keras_predict import KerasPredictor


class Brain(brains.Brains):

    def __init__(self, sensors, actuators):
        super(Brain, self).__init__(sensors, actuators, brain_path=None)
        self.cont = 0
        self.net_w = KerasPredictor('path_to_w')
        self.k_v = 5
        self.motors = self.get_motors('motors_0')

    def load_brain(self, path):
        raise AttributeError("Brain object has no attribute 'load_brain'")

    def execute(self):
        image = self.get_image('camera_0')

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        prediction_w = self.net_w.predict(image)

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(self.k_v)
            self.motors.sendW(prediction_w)
