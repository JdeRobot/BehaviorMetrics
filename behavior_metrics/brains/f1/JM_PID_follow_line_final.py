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
        
        self.kp = 0.006
        self.kd = 0.0075
        self.ki = 0.00005
        self.anterior_error = 0
        self.anterior_u = -1
        self.w_max = 0.5
        self.v_max = 4
        self.v_min = 1.5
        self.vel = 2
        self.inc = 0.20
        
        self.lock = threading.Lock()
        

    def execute(self):
        try:
            img = self.camera.getImage().data
            centro = img.shape[1] / 2
            # LECTURA DE IMAGEN Y PROCESADO
            img = self.camera.getImage().data
            image_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #value_min_HSV = np.array([0, 100, 100])
            #value_max_HSV = np.array([179, 255, 255])
            value_min_HSV = np.array([0, 50, 50])
            value_max_HSV = np.array([180, 255, 255])
            mask = cv2.inRange(image_HSV, value_min_HSV, value_max_HSV)
            mask[275:, :] = 0
            #print(sum(sum(mask)))
            # COMPROBAR SI VE LA LÍNEA
            if sum(sum(mask)) < 4500:
                self.motors.sendW(self.anterior_u)
                self.motors.sendV(0)
                print("he perdido la línea, la estoy buscando...")

            else:
                moment = cv2.moments(mask)
                X = int(moment["m10"] / (moment["m00"] + 1e-5))
                Y = int(moment["m01"] / (moment["m00"] + 1e-5))

                #cv2.circle(img, (X, Y), 10, (0, 255, 0), 2)
                #cv2.line(img, (X, Y), (centro, Y), (255, 0, 0), 2)
                error = X - centro
                diferencia = error - self.anterior_error
                sumatorio = self.anterior_error + error

                # CONTROL VELOCIDAD LINEAL VARIABLE
                if abs(error) > 50:
                    if (self.vel - 5 * self.inc) >= self.v_min:
                        self.vel -= 5 * self.inc
                else:
                    if (self.vel + self.inc) <= self.v_max:
                        self.vel += self.inc

                # VEL CONSTANTE:#HAL.motors.sendV(2.5)

                # CONTROL VELOCIDAD ANGULAR
                u = ((-self.kp * error) - (self.ki * sumatorio) - (self.kd * diferencia))

                if u > self.w_max:
                    u = self.w_max  # -0.1
                elif u < (-self.w_max):
                    u = (-self.w_max)
                self.motors.sendW(u)
                self.motors.sendV(self.vel)

                # ACTUALIZACIÓN Y MOSTRADO IMAGEN
                self.anterior_error = error
                self.anterior_u = u

        except Exception as err:
            print(err)


    