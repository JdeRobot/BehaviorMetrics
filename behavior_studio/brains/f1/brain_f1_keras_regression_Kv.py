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
        self.net_w = KerasPredictor('path_to_w')
        self.k_v = 5

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
        prediction_w = self.net_w.predict(image, type='regression')

        if prediction_w != '' and prediction_w != '':
            self.motors.sendV(self.k_v)
            self.motors.sendW(prediction_w)

        self.update_frame('frame_0', image)
