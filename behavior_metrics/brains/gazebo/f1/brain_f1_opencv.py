#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import cv2
import math
import numpy as np
import threading
import time
from albumentations import (
    Compose, Normalize, RandomRain, RandomBrightness, RandomShadow, RandomSnow, RandomFog, RandomSunFlare
)
from utils.constants import DATASETS_DIR, ROOT_PATH

error = 0
integral = 0
v = 0
w = 0
current = 'straight'
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
MAGENTA = (255, 0, 255)
TH = 500
buf = np.ones(25)
V = 4
V_CURVE = 3.5
V_MULT = 2
v_mult = V_MULT

GENERATED_DATASETS_DIR = ROOT_PATH + '/' + DATASETS_DIR


class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config

        self.threshold_image = np.zeros((640, 360, 3), np.uint8)
        self.color_image = np.zeros((640, 360, 3), np.uint8)
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()
        self.color_image_lock = threading.Lock()
        self.cont = 0
        self.iteration = 0

        # self.previous_timestamp = 0
        # self.previous_image = 0

        self.previous_v = None
        self.previous_w = None
        self.previous_w_normalized = None
        self.suddenness_distance = []

        # Save dataset
        '''
        header = ['image_name', 'v', 'w', 'timestamp']
        with open(GENERATED_DATASETS_DIR + 'difficult_situations_01_06_2022/many_curves_4/data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        '''
        time.sleep(2)

    def update_frame(self, frame_id, data, current_angular_speed=None, previous_angular_speed=None, distance=None):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        if current_angular_speed:
            data = np.array(data, copy=True)

            x1, y1 = int(data.shape[:2][1] / 2), data.shape[:2][0]  # ancho, alto
            length = 200
            angle = (90 + int(math.degrees(-current_angular_speed))) * 3.14 / 180.0
            x2 = int(x1 - length * math.cos(angle))
            y2 = int(y1 - length * math.sin(angle))

            line_thickness = 10
            cv2.line(data, (x1, y1), (x2, y2), (0, 0, 0), thickness=line_thickness)
            length = 150
            angle = (90 + int(math.degrees(-previous_angular_speed))) * 3.14 / 180.0
            x2 = int(x1 - length * math.cos(angle))
            y2 = int(y1 - length * math.sin(angle))

            cv2.line(data, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness)
            if float(distance) > 0.01:
                cv2.putText(data, distance, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(data, distance, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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
            current = 'curve'
        else:
            buf = np.roll(buf, 1)
            buf[0] = 1
            if np.all(buf == 1):
                current = 'straight'
        return (l1, l2)

    def getPoint(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right) / 2 + left
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
        # red_lower = (0, 255, 15)
        red_lower = (0, 110, 15)
        # kernel = np.ones((8, 8), np.uint8)
        '''
        if type(self.previous_image) == int:
            self.previous_image = self.camera.getImage().data
            self.previous_timestamp = timestamp
        if (timestamp - self.previous_timestamp >= 0.085):
            self.previous_image = self.camera.getImage().data
        image = self.previous_image
        '''

        image = self.camera.getImage().data
        if image.shape == (3, 3, 3):
            time.sleep(3)

        '''
        save_dataset = False
        if (timestamp - self.previous_timestamp  >= 0.085):
        #if (timestamp - self.previous_timestamp  >= 0.045):
            #print(timestamp)
            self.previous_timestamp = timestamp
            save_dataset = True
            # Save dataset
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(GENERATED_DATASETS_DIR + 'difficult_situations_01_06_2022/many_curves_4/' + str(self.iteration) + '.png', rgb_image)
        '''
        image = self.handler.transform_image(image, self.config['ImageTranform'])
        # Save dataset
        # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(GENERATED_DATASETS_DIR + 'montreal_12_05_2022_opencv_anticlockwise_1/' + str(self.iteration) + '.png', rgb_image)
        image_cropped = image[230:, :, :]
        image_blur = cv2.GaussianBlur(image_cropped, (27, 27), 0)

        image_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
        image_mask = cv2.inRange(image_hsv, red_lower, red_upper)
        # image_eroded = cv2.erode(image_mask, kernel, iterations=3)

        rows, cols = image_mask.shape
        rows = rows - 1  # para evitar desbordamiento

        alt = 0
        ff = cv2.reduce(image_mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
        if np.count_nonzero(ff[:, 0]) > 0:
            alt = np.min(np.nonzero(ff[:, 0]))

        points = []
        for i in range(3):
            if i == 0:
                index = alt
            else:
                index = rows // (2 * i)
            points.append((self.getPoint(index, image_mask), index))

        points.append((self.getPoint(rows, image_mask), rows))

        l, l2 = self.detect(points)

        if current == 'straight':
            kp = 0.001
            kd = 0.004
            ki = 0
            cv2.circle(image_mask, (0, cols // 2), 6, RED, -1)

            if image_cropped[0, cols // 2, 0] < 170 and v > 8:
                accel = -0.4
            else:
                accel = 0.3

            v_mult = v_mult + accel
            if v_mult > 6:
                v_mult = 6
            v = V * v_mult
        else:
            kp = 0.011  # 0.018
            kd = 0.011  # 0.011
            ki = 0
            v_mult = V_MULT
            v = V_CURVE * v_mult

        new_error = cols // 2 - points[0][0]

        proportional = kp * new_error
        error_diff = new_error - error
        error = new_error
        derivative = kd * error_diff
        integral = integral + error
        integral = ki * integral

        w = proportional + derivative + integral
        self.motors.sendW(w)
        self.motors.sendV(v)

        self.update_frame('frame_0', image)
        current_w_normalized = w

        v = np.interp(np.array([v]), (6.5, 24), (0, 1))[0]
        w = np.interp(np.array([w]), (-7.1, 7.1), (0, 1))[0]
        if self.previous_v != None:
            a = np.array((v, w))
            b = np.array((self.previous_v, self.previous_w))
            distance = np.linalg.norm(a - b)
            self.suddenness_distance.append(distance)
        self.previous_v = v
        self.previous_w = w

        if self.previous_w_normalized != None:
            self.update_frame('frame_2', image, current_w_normalized, self.previous_w_normalized, str(round(distance, 4)))
        self.previous_w_normalized = current_w_normalized

        '''
        if (save_dataset):
            # Save dataset
            iteration_data = [str(self.iteration) + '.png', v, w, self.previous_timestamp]
            with open(GENERATED_DATASETS_DIR + 'difficult_situations_01_06_2022/many_curves_4/data.csv', 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(iteration_data)
            print(self.iteration)
            self.iteration += 1
        '''
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
