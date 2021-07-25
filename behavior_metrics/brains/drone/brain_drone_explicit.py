#!/usr/bin/env python

from numpy.lib.type_check import imag
from drone_wrapper import DroneWrapper
import numpy as np
import threading
import time
import cv2
from collections import deque
#from utils.constants import PRETRAINED_MODELS_DIR, ROOT_PATH
from os import path
import json

#SAVE_DIR = ROOT_PATH + '/' + PRETRAINED_MODELS_DIR + 'drone_models/'

TARGET_HEIGHT = 0.8
TARGET_LANE_WIDTH = 0.06

class Brain:

    def __init__(self, handler=None, config=None):
        self.drone = DroneWrapper()
        self.handler = handler
        # self.drone.takeoff()
        self.takeoff = False

        self.speed_history = deque([], maxlen=100)
        self.speedz_history = deque([0]*100, maxlen=100)
        self.rot_history = deque([], maxlen=1)

        self.x_middle_left_above = 0
        self.deviation_left = 0
        self.iteration = 0
        self.initial_height_reached = False
        self.json_data = []
        self.lock = threading.Lock()
        self.last_lane_width = 0.5

    def update_frame(self, frame_id, data):
        self.handler.update_frame(frame_id, data)

    def getPose3d(self):
        return self.drone.get_position()

    def check_center(self, position_x):
        if (len(position_x[0]) > 1):
            x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
            not_found = False
        else:
            # The center of the line is in position 326
            x_middle = 326
            not_found = True
        return x_middle, not_found


    def exception_case(self, x_middle_left_middle, deviation):
        dif = x_middle_left_middle - self.x_middle_left_above

        if (abs(dif) < 80):
            rotation = -(0.008 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif (abs(dif) < 130):
            rotation = -(0.0075 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif (abs(dif) < 190):
            rotation = -(0.007 * deviation + 0.0005 * (deviation - self.deviation_left))
        else:
            rotation = -(0.0065 * deviation + 0.0005 * (deviation - self.deviation_left))

        speed = 2
        return speed, rotation


    def straight_case(self, deviation, dif):
        if (abs(dif) < 35):
            rotation = -(0.0034 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 2
        elif (abs(dif) < 90):
            rotation = -(0.0032 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 1.5
        else:
            rotation = -(0.0029 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 1

        return speed, rotation

    def curve_case(self, deviation, dif):
        if (abs(dif) < 50):
            rotation = -(0.001 * deviation + 0.0006 * (deviation - self.deviation_left))
        if (abs(dif) < 80):
            rotation = -(0.0095 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif (abs(dif) < 130):
            rotation = -(0.0090 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif (abs(dif) < 190):
            rotation = -(0.0085 * deviation + 0.0005 * (deviation - self.deviation_left))
        else:
            rotation = -(0.0075 * deviation + 0.0005 * (deviation - self.deviation_left))

        speed = 1.0
        return speed, rotation
    
    def get_point(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right)/2 + left
        return int(mid)

    def calculate_line_mask(self, white_region, ncols):

        lines = []
        white_line = []
        for i in range(white_region.shape[0]):
            white_line.append(white_region[i])
            if i < white_region.shape[0]-1:
                if abs(white_region[i] - white_region[i+1]) > 10:
                    lines.append(white_line)
                    white_line = []
        lines.append(white_line)

        if len(lines) == 2:
            middle_line = np.argmin([len(line) for line in lines])
            white_region = lines[middle_line]
        elif len(lines) == 3:
            white_region = lines[1]
        elif len(lines) == 1:
            white_region = lines[0]
        else:
            middle_line = np.argmin([len(line) for line in lines])
            white_region = lines[middle_line]

        return white_region

    def execute(self):

        if self.iteration == 0 and not self.takeoff:
            self.drone.takeoff()
            self.takeoff = True
            self.initial_flight_done = False

        img_frontal = self.drone.get_frontal_image()
        img_ventral = self.drone.get_ventral_image()
        # Both the above images are cv2 images

        if img_frontal.shape == (3, 3, 3) or img_ventral.shape == (3, 3, 3):
            time.sleep(3)
            
        self.update_frame('frame_0', img_frontal)
        self.update_frame('frame_1', img_ventral)
        
        image = img_frontal
        
        try:
            #cv2.imwrite(SAVE_DIR + 'many_curves_data/Images/image{}.png'.format(self.iteration), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #print('written many_curves_data/Images/image{}.png'.format(self.iteration))
            image_cropped = image[230:, :, :]
            image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,50,50])
            upper_red = np.array([180,255,255])
            image_mask = cv2.inRange(image_hsv, lower_red, upper_red)

            # show image in gui -> frame_0
            
            rows, cols = image_mask.shape
            rows = rows - 1     # para evitar desbordamiento

            height_mask = image_mask[9:,:]/255
            white_region = np.where(height_mask[0,:]==1)[0]

            if white_region.shape[0] > 0:
                line_mask = self.calculate_line_mask(white_region, cols)
                lane_width = len(line_mask)/cols
                self.last_lane_width = lane_width
            else:
                lane_width = self.last_lane_width

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
                points.append((self.get_point(index, image_mask), index))

            points.append((self.get_point(rows, image_mask), rows))

            # We convert to show it
            # Shape gives us the number of rows and columns of an image
            size = image_mask.shape
            rows = size[0]
            columns = size[1]

            # We look for the position on the x axis of the pixels that have value 1 in different positions and
            position_x_down = np.where(image_mask[points[3][1], :])
            position_x_middle = np.where(image_mask [points[1][1], :])
            position_x_above = np.where(image_mask[points[2][1], :])        

            # We see that white pixels have been located and we look if the center is located
            # In this way we can know if the car has left the circuit
            x_middle_left_down, not_found_down = self.check_center(position_x_down)
            x_middle_left_middle, not_found_middle = self.check_center(position_x_middle)

            # We look if white pixels of the row above are located
            if (len(position_x_above[0]) > 1):
                self.x_middle_left_above = (position_x_above[0][0] + position_x_above[0][len(position_x_above[0]) - 1]) / 2
                # We look at the deviation from the central position. The center of the line is in position cols/2
                deviation = self.x_middle_left_above - (cols/2)

                # If the row below has been lost we have a different case, which we treat as an exception
                if not_found_down == True:
                    speed, rotation = self.exception_case(x_middle_left_middle, deviation)
                    lane_width = 0.5 #self.last_lane_width
                else:
                    # We check is formula 1 is in curve or straight
                    dif = x_middle_left_down - self.x_middle_left_above
                    x = float(((-dif) * (310 - 350))) / float(260-350) + x_middle_left_down

                    if abs(x - x_middle_left_middle) < 2:
                        speed, rotation = self.straight_case(deviation, dif)
                    else:
                        speed, rotation = self.curve_case(deviation, dif)

                # We update the deviation
                self.deviation_left = deviation
            else:
                # If the formula 1 leaves the red line, the line is searched
                if self.x_middle_left_above > (columns/2):
                    rotation = -1
                else:
                    rotation = 1
                speed = 0.1
                lane_width = self.last_lane_width

            pitch = self.drone.get_pitch()

            if abs(self.getPose3d()[2] - TARGET_HEIGHT) > 0.2 or not self.initial_flight_done:
                speed = 0
                rotation = 0
                curr_vel_z = self.drone.get_velocity()[2]
                speed_z = -(self.getPose3d()[2] - TARGET_HEIGHT) - 0.3*curr_vel_z
                self.initial_flight_done = True
            else:
                curr_vel_z = self.drone.get_velocity()[2]
                speed_z = -3*(TARGET_LANE_WIDTH - lane_width) - 0.2*curr_vel_z

            # print("Status: Height->{} | Observed Lane Width->{} | Z-vel->{} | cmd_vel_z->{} | pitch->{}".format(
            #     self.getPose3d()[2], lane_width, curr_vel_z, speed_z, np.degrees(pitch)))

            self.speed_history.append(speed)
            self.speedz_history.append(speed_z)
            self.rot_history.append(rotation)

            speed_cmd = np.mean(self.speed_history)
            speed_z_cmd = np.clip(speed_z,-2,2)
            rotation_cmd = np.mean(self.rot_history)

            #self.json_data.append({'iter': self.iteration, 'v': speed_cmd, 'w': rotation, 'vz': speed_z_cmd})
            #with open(SAVE_DIR + 'many_curves_data/data.json', 'w') as outfile:
            #    json.dump(self.json_data, outfile)

            self.drone.set_cmd_vel(speed_cmd, 0, speed_z_cmd, rotation_cmd)

            self.iteration += 1
            
        except Exception as err:
            print(err)
