#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
from datetime import datetime

import math
import jderobot
import cv2
import numpy as np

from interfaces.motors import PublisherMotors


time_cycle = 80


# Show text into an image
font = cv2.FONT_HERSHEY_SIMPLEX

witdh = 640
mid = 320

last_center_line = 0

# Constantes de giro - kp más alta corrige más
kp = 0.02       ## valores para 20 m/s --> 0.019
kd = 0.012        ## valores para 20 m/s --> 0.011
last_error = 0

# Constantes de Velocidad
kpv = 0.01    ## valores para 20 m/s --> 0.09
kdv = 0.03   ## valores para 20 m/s --> 0.003
vel_max = 20  ## probado con 20 m/s
last_vel = 0


class MyAlgorithm(threading.Thread):

    def __init__(self, camera, motors, network):
        self.camera = camera
        self.motors = motors
        self.network = network
        self.threshold_image = np.zeros((640,360,3), np.uint8)
        self.color_image = np.zeros((640,360,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
        self.cont = 0
    
    def setNetwork(self, custom_net):
        self.network = custom_net
        self.network.setCamera(self.camera)
    
    def getImage(self):
        self.lock.acquire()
        img = self.camera.getImage().data
        self.lock.release()
        return img

    def set_color_image (self, image):
        img  = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.color_image_lock.acquire()
        self.color_image = img
        self.color_image_lock.release()
        
    def get_color_image (self):
        self.color_image_lock.acquire()
        img = np.copy(self.color_image)
        self.color_image_lock.release()
        return img
        
    def set_threshold_image (self, image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
        
    def get_threshold_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.algorithm()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def net_classification_7w_5v(self, prediction_v, prediction_w):

        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND 5 FOR V
        if prediction_v == "slow":
            self.motors.sendV(5)
        elif prediction_v == "moderate":
            self.motors.sendV(8)
        elif prediction_v == "fast":
            self.motors.sendV(10)
        elif prediction_v == "very_fast":
            self.motors.sendV(13)
        elif prediction_v == 'negative':
            self.motors.sendV(-0.6)
        
        if prediction_w == "radically_left":
            self.motors.sendW(1.7)
        elif prediction_w == "moderately_left":
            self.motors.sendW(0.75)
        elif prediction_w == "slightly_left":
            self.motors.sendW(0.25)
        elif prediction_w == "slight":
            self.motors.sendW(0)
        elif prediction_w == "slightly_right":
            self.motors.sendW(-0.25)
        elif prediction_w == "moderately_right":
            self.motors.sendW(-0.75)
        elif prediction_w == "radically_right":
            self.motors.sendW(-1.7)

    def net_classification_7w_4v(self, prediction_v, prediction_w):
        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND 4 FOR V
        if prediction_v == "slow":
            self.motors.sendV(5)
        elif prediction_v == "moderate":
            #self.motors.sendV(6)
            self.motors.sendV(8)
        elif prediction_v == "fast":
            #self.motors.sendV(7)
            self.motors.sendV(10)
        elif prediction_v == "very_fast":
            #self.motors.sendV(8)
            self.motors.sendV(13)
        
        if prediction_w == "radically_left":
            self.motors.sendW(1.9)
        elif prediction_w == "moderately_left":
            self.motors.sendW(0.75)
            #self.motors.sendW(0.75)
        elif prediction_w == "slightly_left":
            self.motors.sendW(0.25)
            #self.motors.sendW(0.5)
        elif prediction_w == "slight":
            self.motors.sendW(0)
        elif prediction_w == "slightly_right":
            self.motors.sendW(-0.25)
            #self.motors.sendW(-0.5)
        elif prediction_w == "moderately_right":
            self.motors.sendW(-0.75)
            #self.motors.sendW(-0.75)
        elif prediction_w == "radically_right":
            self.motors.sendW(-1.9)

    def net_classification_7w_constant_v(self, prediction_w):
        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND CONSTANT V
        self.motors.sendV(5)
        
        if prediction_w == "radically_left":
            self.motors.sendW(1.8)
        elif prediction_w == "moderately_left":
            self.motors.sendW(0.75)
        elif prediction_w == "slightly_left":
            self.motors.sendW(0.25)
        elif prediction_w == "slight":
            self.motors.sendW(0)
        elif prediction_w == "slightly_right":
            self.motors.sendW(-0.25)
        elif prediction_w == "moderately_right":
            self.motors.sendW(-0.75)
        elif prediction_w == "radically_right":
            self.motors.sendW(-1.8)
    
    def net_regression(self, prediction_v, prediction_w):
        # REGRESSION NETWORK FOR W AND V
        self.motors.sendV(prediction_v)
        self.motors.sendW(prediction_w)

    def net_regression_constant_v(self, prediction_v, prediction_w):
        # REGRESSION NETWORK FOR W AND V
        self.motors.sendV(3)
        self.motors.sendW(prediction_w)

    def algorithm(self):
        #GETTING THE IMAGES
        image = self.getImage()

        if self.cont > 0:
            print("Runing...")
            self.cont += 1

        prediction_v = self.network.prediction_v
        prediction_w = self.network.prediction_w

        if prediction_w != '' and prediction_w != '':

            net_type = self.network.__class__.__name__
            if net_type == 'ClassificationNetwork':
                # self.net_classification_7w_constant_v(prediction_w)
                # self.net_classification_7w_4v(prediction_v, prediction_w)
                self.net_classification_7w_5v(prediction_v, prediction_w)
            elif net_type == 'RegressionNetwork':
                # self.net_regression_constant_v(prediction_w)
                self.net_regression(prediction_v, prediction_w)
       
        #SHOW THE FILTERED IMAGE ON THE GUI
        self.set_threshold_image(image)
    








    ################################################
    ###### MANUAL SOLUTION from MOVA
    ####################################################

    # def processed_image(self, img):
        
    #     img = img[220:]
    #     img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(img_proc, (1, 235, 60), (180, 255, 255))
    #     wall = img[12][320][0]
    #     line = mask[30,:]
    #     base = mask[250,:]
            
    #     try:
    #         line_center = np.divide(np.max(np.nonzero(line)) - np.min(np.nonzero(line)), 2)
    #         line_center = np.min(np.nonzero(line)) + line_center
    #     except ValueError:
    #         line_center = last_center_line
            
    #     # Puntos centrales de la línea segmentada
    #     cv2.line(img, (line_center, 30), (line_center, 30), (255, 255, 255), thickness=5)
        
    #     cv2.line(img, (320, 30),  (320, 30),  (0, 255, 0), thickness=5)
    #     cv2.line(img, (320, 12),  (320, 12),  (255, 255, 0), thickness=5)

    #     cv2.line(img, (320, 30), (line_center, 30),  (255, 0, 0), thickness=2)
        
    #     return mask, line_center, wall



    # # EXECUTE
    # def algorithm(self):
        
    #     img = self.getImage()
    #     img_proc, line_center, wall = self.processed_image(img)
    #     error_line = np.subtract(mid, line_center).item()


    #     global last_error
    #     global vel_max
    #     global last_vel
        
    #     giro = kp * error_line + kd * (error_line - last_error)
    #     self.motors.sendW(giro)

    #     vel_error = kpv * abs(error_line) + abs(kdv * (error_line - last_error))
        
    #     if abs(error_line) in range(0, 15) and wall >= 178:
    #         self.motors.sendV(vel_max)
    #     elif wall in range(0,179):
    #         if wall < 50:
    #             brake = 10
    #         else:
    #             brake = 5
    #         vel_correccion = abs(vel_max - vel_error - brake)
    #         self.motors.sendV(vel_correccion)
    #     elif wall == 0:
    #         vel_correccion = abs(vel_max - vel_error - (2 * brake))
    #         self.motors.sendV(vel_correccion)
    #     else:
    #         pass

    #     last_error = error_line
    #     last_vel = vel_max

    #     self.set_threshold_image(img_proc)
