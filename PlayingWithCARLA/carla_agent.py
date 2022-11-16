from carla.agent.agent import Agent
from carla.client import VehicleControl

class ForwardAgent(Agent):

    def run_step(self, measurements, sensor_data, directions, target):
    """
    Function to run a control step in the CARLA vehicle.
    """
    control = VehicleControl()
    control.throttle = 0.9
    return control
