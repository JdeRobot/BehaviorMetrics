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
        #self.json_data = []


        # Inicializar variables
        self.consigna = 320
        self.error_prev = 0
        self.error = 0
        self.u = 0
        self.v = 3.6

        # Constantes del controlador
        self.Kp = 0.0045
        self.Kd = 0.003




        #header = ['image_name', 'v', 'w']
        #with open(GENERATED_DATASETS_DIR + 'only_curves_25_03_2022/extended_simple_1/data.csv', 'w', encoding='UTF8') as f:
        #    writer = csv.writer(f)
        #    writer.writerow(header)

        self.lock = threading.Lock()



    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.
        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def calculate_mask(self, img):
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Umbrales para el rojo en el espacio HSV
        #lower_red1 = np.array([170, 100, 100])
        #upper_red1 = np.array([179, 255, 255])

        #lower_red2 = np.array([0, 100, 100])
        #upper_red2 = np.array([10, 255, 255])

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask = cv2.inRange(img_HSV, lower_red, upper_red)

        # Máscara(imagen binarizada)
        #mask1 = cv2.inRange(img_HSV, lower_red1, upper_red1)
        #mask2 = cv2.inRange(img_HSV, lower_red2, upper_red2)
        #mask = mask1 + mask2
        return mask


    def calculate_roi_mask(self, mask, y_min, y_max):
        roi_mask = mask.copy()
        roi_mask[:y_min, :] = 0
        roi_mask[y_max:, :] = 0
        return roi_mask

    def execute(self):

        try:
            img = self.camera.getImage().data
            mask = self.calculate_mask(img)
            roi = self.calculate_roi_mask(mask, 240, 255)

            #_, contours_roi, hierarchy_roi = cv2.findContours(roi, 1, cv2.CHAIN_APPROX_NONE)
            contours_roi, hierarchy_roi = cv2.findContours(roi, 1, cv2.CHAIN_APPROX_NONE)

            if len(contours_roi) > 0:

                # Se calcula el punto en el horizonte
                p_horizonte = np.argwhere(mask == 255)[0]

                # Nos quedamos con el contorno de área máxima
                c = max(contours_roi, key=cv2.contourArea)
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    desviacion = np.abs(cx - p_horizonte[1])
                    if desviacion <= 10:  # Recta
                        estado = 'Recta'
                        self.v = self.v * 1.04
                        print(self.v)
                        if self.v > 5:
                            self.v = 5
                    elif desviacion > 10 and desviacion <= 55:  # Curva suave
                        estado = 'Curva suave'
                        self.v = self.v * 1.01
                        if self.v > 3.4:
                            self.v = 3.4
                    else:  # Curva cerrada
                        estado = 'Curva cerrada'
                        self.v = self.v * 0.3
                        if self.v < 2.6:
                            self.v = 2.6

                    # Calculo del error y de la velocidad angular(regulador PD)
                    self.error = self.consigna - cx
                    dedt = self.error - self.error_prev
                    w = self.Kp * self.error + self.Kd * dedt

                    # Se guarda el error actual para la iteración siguiente
                    self.error_prev = self.error

                    # Se mandan la velocidad lineal y angular a los motores
                    self.motors.sendW(w)
                    self.motors.sendV(self.v)

            else:
                # Modo búsqueda de línea
                self.motors.sendW(0.1)
                self.motors.sendV(0)
                self.v = 0.3
                estado = 'Buscando la linea'

        except Exception as err:
            print(err)