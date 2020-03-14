#!/usr/bin/python
#-*- coding: utf-8 -*-
from brains import Brains

class Brain(Brains):

    def __init__(self, sensors, actuators):
        super(Brain, self).__init__(sensors, actuators, brain_path=None)
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')

     
    def execute(self):
        self.motors.sendV(0)
        self.motors.sendW(0.2)