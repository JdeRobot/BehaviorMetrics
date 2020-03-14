"""
    Robot: F1
    Framework: keras
    Number of networks: 2
    Network type: Classification
    Predicionts:
        linear speed(v)
        angular speed(w)

    This brain uses a classification network based on Keras framework to predict the linear and angular velocity
    of the F1 car. For that task it uses two different classification convolutional neural networks, one for v
    (with 4 classes) and another one for w (with 7 classes)

"""

from brains import Brains
from behaviorlib.keras.keras_predict import KerasPredictor


class Brain(Brains):

    def __init__(self, sensors, actuators):
        super(Brain, self).__init__(sensors, actuators, brain_path=None)
        self.cont = 0
        self.net_v = KerasPredictor('path_to_v')
        self.net_w = KerasPredictor('path_to_w')
        self.motors = self.get_motors('motors_0')

    def load_brain(self, path):
        raise AttributeError("Brain object has no attribute 'load_brain'")
   
    def calculate_v(self, predicted_class):
        """
        Method that calculates the linear speed of the robot (v) based on the predicted class

        The class-speed label conversion for v is as follows:
            class 0 = slow
            class 1 = moderate
            class 2 = fast
            class 3 = very fast
        """

        if predicted_class == 0:
            self.motors.sendV(5)
        elif predicted_class == 1:
            self.motors.sendV(8)
        elif predicted_class == 2:
            self.motors.sendV(10)
        elif predicted_class == 3:
            self.motors.sendV(13)

    def calculate_w(self, predicted_class):
        """
        Method that calculates the linear speed of the robot (v) based on the predicted class

        The class-speed label conversion for v is as follows:
            class 0 = radically left
            class 1 = moderate left
            class 2 = slightly left
            class 3 = slight
            class 4 = slightly right
            class 5 = moderate right
            class 6 = radically right
        """
        if predicted_class == 0:
            self.motors.sendW(1.7)
        elif predicted_class == 1:
            self.motors.sendW(0.75)
        elif predicted_class == 2:
            self.motors.sendW(0.25)
        elif predicted_class == 3:
            self.motors.sendW(0)
        elif predicted_class == 4:
            self.motors.sendW(-0.25)
        elif predicted_class == 5:
            self.motors.sendW(-0.75)
        elif predicted_class == 6:
            self.motors.sendW(-1.7)

    def execute(self):
        image = self.get_image('camera_0')

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        prediction_v = self.net_v.predict(image)
        prediction_w = self.net_w.predict(image)

        if prediction_w != '' and prediction_w != '':
            v = self.calculate_v(prediction_v)
            w = self.calculate_w(prediction_w)
            self.motors.sendV(v)
            self.motors.sendW(w)
