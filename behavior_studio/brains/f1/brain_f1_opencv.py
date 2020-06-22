#!/usr/bin/python
# -*- coding: utf-8 -*-
import threading
import time

import cv2
import numpy as np

time_cycle = 80
error = 0
integral = 0
v = 0
w = 0
current = 'recta'
time_cycle = 80
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
TH = 500
buf = np.ones(25)
V = 4
V_MULT = 2
v_mult = V_MULT


class Brain:

    def __init__(self, sensors, actuators, handler):
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.cont = 0
        time.sleep(2)

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def collinear3(self, x1, y1, x2, y2, x3, y3):
        line = 0
        line = np.abs((y1 - y2) * (x1 - x3) - (y1 - y3) * (x1 - x2))
        return line

    def detect(self, points):
        global current
        global buf
        l2 = 0
        l1 = self.collinear3(points[0][1], points[0][0], points[1][1], points[1][0], points[2][1], points[2][0])
        if l1 > TH:
            buf[0] = 0
            current = 'curva'
        else:
            buf = np.roll(buf, 1)
            buf[0] = 1
            if np.all(buf == 1):
                current = 'recta'
        return (l1, l2)

    def getPoint(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right)/2 + left
        return int(mid)

    def execute(self):
        global error
        global integral
        global current
        global v_mult
        global v
        global w
        red_upper = (179, 255, 255)
        # red_lower=(0,255,171)
        red_lower = (0, 255, 15)
        # kernel = np.ones((8, 8), np.uint8)
        image = self.camera.getImage().data
        if image.shape == (3, 3, 3):
            time.sleep(3)

        image_cropped = image[230:, :, :]
        image_blur = cv2.GaussianBlur(image_cropped, (27, 27), 0)
        image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
        image_mask = cv2.inRange(image_hsv, red_lower, red_upper)
        # image_eroded = cv2.erode(image_mask, kernel, iterations=3)

        # show image in gui -> frame_0
        self.update_frame('frame_0', image)

        rows, cols = image_mask.shape
        rows = rows - 1     # para evitar desbordamiento

        alt = 0
        ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        if np.count_nonzero(ff[:, 0]) > 0:
            alt = np.min(np.nonzero(ff[:, 0]))

        points = []
        for i in range(3):
            if i == 0:
                index = alt
            else:
                index = rows//(2*i)
            points.append((self.getPoint(index, image_mask), index))

        points.append((self.getPoint(rows, image_mask), rows))

        l, l2 = self.detect(points)

        if current == 'recta':
            kp = 0.001
            kd = 0.004
            ki = 0
            cv2.circle(image_mask, (0, cols//2), 6, RED, -1)

            if image_cropped[0, cols//2, 0] < 170 and v > 8:
                accel = -0.4
            else:
                accel = 0.3

            v_mult = v_mult + accel
            if v_mult > 6:
                v_mult = 6
        else:
            kp = 0.011  # 0.018
            kd = 0.011  # 0.011
            ki = 0
            v_mult = V_MULT

        new_error = cols//2 - points[0][0]

        proportional = kp * new_error
        error_diff = new_error - error
        error = new_error
        derivative = kd * error_diff
        integral = integral + error
        integral = ki * integral

        w = proportional + derivative + integral
        v = V * v_mult
        self.motors.sendW(w)
        self.motors.sendV(v)

        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)
        cv2.circle(image_mask, points[0], 6, GREEN, -1)
        cv2.circle(image_mask, points[1], 6, GREEN, -1)  # punto central rows/2
        cv2.circle(image_mask, points[2], 6, GREEN, -1)
        cv2.circle(image_mask, points[3], 6, GREEN, -1)
        cv2.putText(image_mask, 'w: {:+.2f} v: {}'.format(w, v),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(image_mask, 'collinearU: {} collinearD: {}'.format(l, l2),
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2, cv2.LINE_AA)
        cv2.putText(image_mask, 'actual: {}'.format(current),
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, MAGENTA, 2, cv2.LINE_AA)

        self.update_frame('frame_1', image_mask)
