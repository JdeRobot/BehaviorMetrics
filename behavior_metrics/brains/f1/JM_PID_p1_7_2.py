#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import threading
import time
import cv2
import csv

from utils.constants import DATASETS_DIR, ROOT_PATH
GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR

class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler

        self.x_middle_left_above = 0
        self.deviation_left = 0
        self.iteration = 0

        self.wheel = 0
        self.pedal = 0
        self.kpw = 0.00535
        self.kdw = 0.00956
        self.kiw = 0.000003
        self.kpp = 1.4
        self.speed = 0
        self.speedLimit = 3.4
        self.kps = 0.0001
        self.midW = 323
        self.rows = [245, 258]
        self.thres = 200
        self.lastWheel = 0
        self.lastError = 0
        self.lastInt = 0
        self.isReady = False

        self.lock = threading.Lock()

    def updateError(self, _error):
        self.lastError = _error

    def thereIsLine(self, _img):
        """
        If there is no line at the beginning turn the car.
        Return True if there is line and False in other case and turn step
        """
        _turnStep = 0.1
        _error, _ = self.imgProc(_img, init=True)

        if _error != -self.midW:
            return True, 0
        else:
            return False, _turnStep

    # Sensor
    def imgProc(self, _img, init=False):
        """
        Image processing
        Generate the mask, compute the error and return the error.
        """
        # Generate a mask
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
        mask = (_img[:, :, 1] > self.thres) * 1
        histogram = np.sum(mask[self.rows[0]:self.rows[1], :], axis=0)

        # Error processing, getting the centroid
        _error = np.argmax(histogram) - self.midW
        # if errors is -320 maybe it loss the line
        _error = self.lastError if _error == -self.midW and init is False else _error

        _vertical_ref = (_img[:, self.midW - 5:self.midW + 5, 1] > self.thres) * 1
        _mid_long = np.sum(_vertical_ref)

        return _error, _mid_long

    # Actuators
    def steerB(self, _error):
        """
        PID controller
        Controls the steering wheel
        Return the steering wheel position, KP, KI and KD values
        """
        self.lastInt += _error
        kp = -self.kpw * _error
        ki = -self.kiw * self.lastInt
        kd = -self.kdw * (_error - self.lastError)
        self.wheel = kp + ki + kd
        return self.wheel, kp, ki, kd

    def pedalB(self, _wheel, _mid_long):
        """
        P controller adaptation
        Controls the accelerator pedal, only whit steearing wheel info
        Return the accelerator pedal position
        """
        if self.speed < self.speedLimit:
            self.speed = self.speed + self.kps * _mid_long
            self.pedal = self.speed
        else:
            self.pedal = self.speedLimit - self.kpp * np.abs(_wheel - self.lastWheel)
        self.lastWheel = _wheel
        return self.pedal

    def execute(self):
        try:
            # Get image
            img = self.camera.getImage().data
            if self.isReady is False:
                self.isReady, turnStep = self.thereIsLine(img)
                self.motors.sendW(turnStep)
            else:
                # Processing
                error, mid_long = self.imgProc(img)
                wheel, KPW, KIW, KDW = self.steerB(error)
                pedal = self.pedalB(wheel, mid_long)
                self.updateError(error)
                console.print(
                    "Wheel// Errors: " + str(error) + ", KPW: " + str(KPW) + ", KIW: " + str(KIW) + ", KDW: " + str(
                        KDW) + ", wheel: " + str(wheel) + ", pedal: " + str(pedal))
                # Action
                self.motors.sendW(wheel)
                self.motors.sendV(pedal)

                # GUI verbose
                img[self.rows[0], :, :] = 255
                img[self.rows[1], :, :] = 255
                img[:, self.midW, :] = 255

        except Exception as err:
            print(err)