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
        
        self.dif = 0
        self.kp = 0.0026
        self.kd = 0.01
        self.ki = 0.00075
        self.v_max = 2.6
        self.kp_v = 0.004
        self.vel = 0
        self.dif_0 = 0
        #self.json_data = []
        
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

    ## Funcion que calcula la mascara de color del camino rojo y el punto guia que seguira el coche
    def masks(self, img_hsv, h1, h2):

        # Se han de importar de forma local los módulos
        import cv2
        import numpy as np
        
        # Valores minimos de la mascara
        lower_red_0 = (0, 50, 50)
        upper_red_0 = (10, 255, 255)
        # Valores maximos de la mascara
        lower_red_1 = (170, 50, 50)
        upper_red_1 = (180, 255, 255)
        # Calculo de mascaras en el rango del color deseado (rojo)
        #mask0 = cv2.inRange(img_hsv[h1:h2, :, :], lower_red_0, upper_red_0)
        #mask1 = cv2.inRange(img_hsv[h1:h2, :, :], lower_red_1, upper_red_1)
        mask0 = cv2.inRange(img_hsv, lower_red_0, upper_red_0)
        mask1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
        mask = mask0 + mask1
        
        #print(img_hsv.shape)
        
        #lower_red = np.array([0, 50, 50])
        #upper_red = np.array([180, 255, 255])
        #mask = cv2.inRange(img_hsv, lower_red, upper_red)
        #self.update_frame('frame_0', mask)
        ## Se eliminan los puntos erróneos detectados mediante dos procesos de erosión y dos de dilatación
        # Kernel utilizado para quitar ruido con erosion y dilatacion
        kernel = np.ones((3, 3), np.uint8)
        # Erosion
        cv2.erode(src=mask, dst=mask, kernel=kernel, iterations=2)
        # Dilatacion
        cv2.dilate(src=mask, dst=mask, kernel=kernel, iterations=2)

        # Calculo de contornos de la mascara
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Inicializacion de variables
        cx = None
        cy = None
        # Calculo del centro de masas del contorno de la mascara
        for c in contours:
            M = cv2.moments(c[:, :, :])
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"]) + h1
            
        return cx, cy

    def execute(self):
        try:
            # Se obtiene la imagen
            frame = self.camera.getImage().data
            self.update_frame('frame_0', frame)
            # Se pasan los colores de los frames a HSV (por defecto eran BGR en OpenCV)
            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Se crea una mascara para detectar el color deseado acudiendo a la funcion masks
            cx_1, cy_1 = self.masks(img_hsv, 250, 280)
            # En el caso de perder el punto el coche podrá actuar de 2 maneras:
            print(cx_1, cy_1)
            if cx_1 == None:
                # En el caso de que el coche no tenga ninguna diferencia calculada se detendrá y girará a la izquierda por defecto
                # Este caso podría darse si el coche comienza girado 90º
                if self.dif == 0:
                    print('No encuentro la línea, voy a girar')
                    self.motors.sendW(0.5)

                # En el caso de que el coche haya perdido el punto de referencia disminuirá su velocidad
                else:
                    print('Dificultades detectando la linea, voy a reducir la velocidad')
                    self.motors.sendV(1.4)
                    self.motors.sendW(self.kp * self.dif + self.kd * (self.dif - self.dif_0) + self.ki * (self.dif + self.dif_0))
            else:
                # Se dibuja el punto en el horizonte de la linea
                cv2.circle(frame, (cx_1, cy_1), 2, (0, 255, 0), 4)
                # Se calcula el ancho y el alto de la imagen
                h, w, _ = frame.shape
                # Se guarda la diferencia anterior
                self.dif_0 = self.dif
                # Se actualiza la diferencia
                self.dif = w / 2 - cx_1
                # Se calcula la velocidad segun las circuntancias de la imagen
                self.vel = self.v_max-abs(self.dif)*self.kp_v
                # Se envian las ordenes a los actuadores

                self.motors.sendV(self.vel)
                self.motors.sendW(self.kp * self.dif + self.kd * (self.dif - self.dif_0) + self.ki * (self.dif + self.dif_0))
            
            self.iteration += 1

        except Exception as err:
            print(err)
