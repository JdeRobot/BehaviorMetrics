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
        
        
        #Antes del bucle fijo ciertas variables
        #Velocidades:
        self.v_max = 4
        self.v_min = 1.5
        #self.v_max = 10
        #self.v_min = 3
        self.y = self.v_max #velocidad inicial

        #Constantes de frenado:
        #El sistema de frenado funciona reduciendo respecto el error (el ángulo).
        #La constante kv1 nos hace que la velocidad minima se alcance cuando la
        #velocidad angular es np.pi/9~0.35
        #La constante kv2 resta a la velocidad máxima respecto a la variación del
        # ángulo en la última iteración.
        self.kv1 = (self.v_max-self.v_min)/(np.pi/9)
        self.kv2 = 1

        #Constantes PID:
        self.kp = 1 #0.6
        self.kd = 0.75 #0.7
        self.ki = 0.05
        self.error_ant = 0
        self.integral = 0
        self.errores = np.array([0])

        #Límites para la selección de color (en HSV):
        # self.lower_red = np.array([0,50,50])
        # self.upper_red = np.array([60,255,255])
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([180, 255, 255])
        
        self.lock = threading.Lock()
        

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def execute(self):
        #Capturamos la imagen. Para obtener las dimensiones.
        img = self.camera.getImage().data
        dimensiones = img.shape #dimensiones = (480, 640)

        #El coche tiene la linea roja en el centro de la imagen, pero la cámara está
        #echada hacia el lado izquierdo del coche.
        #Ajustamos el óptimo de manera experimental
        #El óptimo es la columna en la que querremos situar el centroide.
        optimo = dimensiones[1]/2+15

        #Filas entre las que tenemos en cuenta la linea roja.
        #La teoría es que nos interesa más la parte lejana de la línea roja que
        #la cercana para calcular el centroide (el cual usamos para medir el error).
        #No pongo desde la mitad de la imagen porque en curvas muy cerradas y seguidas
        #el centroide se cambia de sentido antes de terminar la curva en la que está.
        fila1 = int(dimensiones[0]/2 + 10)
        fila2 = int(dimensiones[0]/2 + 30)
        print(fila1, fila2)
        try:
            img = self.camera.getImage().data

            #Pasamos la imagen a HSV para hacer una fácil detección del rojo.
            #Aplicamos los thresholds para el rojo con lo que calculamos una máscara.
            #0 para píxeles fuera de los límites, 255 para los que los cumplen
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img, self.lower_red, self.upper_red)
            #Fijamos a 0 de la fila fijada hacia abajo en la máscara.
            #De esta manera solo vemos una parte de la línea roja. La más lejana.
            mask[:fila1] = 0
            mask[fila2:] = 0
            #Inicio girado, máscara no detectada y centroide no calculados anteriormente
            if np.all(mask==0) and "cx" not in locals():
                y = 0
                x = 0.1
                print("Dentro")
                self.motors.sendV(y)
                self.motors.sendW(x)

            else:
                #Calculamos los contornos para visualizar mejor la máscara
                #Con los momentos calculamos el centroide de la línea roja.
                #Con el centroide medimos el error frente a la posición óptima.
                _, contornos = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                #Calculamos los momentos de la máscara. Calculamos el centroide
                momentos = cv2.moments(mask)
                if momentos["m00"] != 0: #Evitamos posibles momentos de píxeles solitarios.
                    cx = int(momentos["m10"] / momentos["m00"])
                    cy = int(momentos["m01"] / momentos["m00"])
                    #console.print("Centros de masa: CX:"+str(cx)+" CY:"+str(cy))

                #Error de la posición del f1.
                #El error es el ángulo que se forma en la posición entre la recta vertical
                #(óptimo) y el centroide.
                #De esta manera, estamos teniendo no solo la diferencia en el eje X, sino
                #también la distancia en el eje Y a la que nos encotramos del centroide.

                #Error: Negativo necesitamos mover hacia la derecha
                #Positivo necesitamos movernos hacia la izq
                error = optimo - cx
                error = np.arctan2(error, (480-cy))
                #console.print("Angulo: "+str(error))

                #Calculamos la derivada como la diferencia del error actual con la del
                #momento anterior
                dt_error = error-self.error_ant

                #Calculamos la integral de los errores como la media de los errores desde la
                #última vez que el error cambia de signo.
                ind = np.where(self.errores==0)[0][-1]
                integral = np.mean(self.errores[ind:]) #np.trapz(errores)

                #Aplicamos PID para la velocidad angular.
                x = self.kp*error+self.kd*dt_error+self.ki*integral

                #Fijamos la velocidad angular máxima a los siguientes valores:
                x = np.clip(x, -0.60, 0.60)

                #Ajustamos la velocidad vertical con un frenado respecto al ángulo.
                #También ajustamos la velocidad vertical con la derivada del ángulo.
                #Esto hace que tengamos una velocidad vertical suavizada y no a escalones.

                y = self.v_max-self.kv1*np.abs(error)+self.kv2*dt_error
                y = np.clip(y, self.v_min, self.v_max)

                #Pasamos la imagen de vuelta a BGR (opencv) y la mostramos.
                #Enviamos las velocidades calculadas.
                #Nos guardamos el error obtenido en esta iteración como error anterior y lo
                #guardamos en una lista para el cálculo integral.

                #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                #cv2.circle(img, (cx, cy), 5, (0,255,0), -1)
                #cv2.drawContours(img, contornos, -1, (0,255,0), 3)
                #cv2.line(img,(dimensiones[1]/2+15,0), (dimensiones[1]/2+15,dimensiones[0]-1),(0,255,255),3)

                self.motors.sendV(y)
                self.motors.sendW(x)
                self.error_ant = error
                self.errores = np.append(self.errores, error)

        except Exception as err:
            print(err)
