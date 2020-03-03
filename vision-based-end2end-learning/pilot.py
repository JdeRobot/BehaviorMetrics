import time
import datetime
import threading

from robot.sensors import Sensors
from robot.actuators import Actuators
import brains

TIME_CYCLE = 80


class Pilot(threading.Thread):

    def __init__(self, config_data):

        robot = config_data['Behaviors']['Robot']
        sensors_config = robot['Sensors']
        actuators_config = robot['Actuators']

        self.sensors = Sensors(sensors_config)
        self.actuators = Actuators(actuators_config)

    def action(self):
        brains.active_brain.do_action(self.sensors, self.actuators)

    def run(self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.action()
            dt = datetime.now() - start_time

            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            if (ms < TIME_CYCLE):
                time.sleep((TIME_CYCLE - ms) / 1000.0)

    def stop(self):
        self.stop_event.set()

    def play(self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        self.kill_event.set()
