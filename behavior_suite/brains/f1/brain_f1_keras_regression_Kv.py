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
from behaviorlib.keraslib.keras_predict import KerasPredictor


class Brain:

    def __init__(self, sensors, actuators, handler=None):
        self.motors = actuators.get_motor('motors_0')
        self.camera = sensors.get_camera('camera_0')
        self.handler = handler
        self.cont = 0
        self.net_w = KerasPredictor('path_to_w')
        self.k_v = 5

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def execute(self):
        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        image = self.camera.getImage().data
        prediction_w = self.net_w.predict(image)

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(self.k_v)
            self.motors.sendW(prediction_w)

        self.update_frame('frame_0', image)
