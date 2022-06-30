#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import threading
import time
import cv2

from utils.noise import add_noise


class Brain:

    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config

        self.x_middle_left_above = 0
        self.deviation_left = 0
        # self.iteration = 0
        # self.json_data = []
        self.lock = threading.Lock()

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """
        self.handler.update_frame(frame_id, data)

    def check_center(self, position_x):
        if len(position_x[0]) > 1:
            x_middle = (position_x[0][0] + position_x[0][len(position_x[0]) - 1]) / 2
            not_found = False
        else:
            # The center of the line is in position 326
            x_middle = 326
            not_found = True
        return x_middle, not_found

    def exception_case(self, x_middle_left_middle, deviation):
        dif = x_middle_left_middle - self.x_middle_left_above

        if abs(dif) < 80:
            rotation = -(0.008 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif abs(dif) < 130:
            rotation = -(0.0075 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif abs(dif) < 190:
            rotation = -(0.007 * deviation + 0.0005 * (deviation - self.deviation_left))
        else:
            rotation = -(0.0065 * deviation + 0.0005 * (deviation - self.deviation_left))

        speed = 5
        return speed, rotation

    def straight_case(self, deviation, dif):
        if abs(dif) < 35:
            rotation = -(0.0054 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 13
        elif abs(dif) < 90:
            rotation = -(0.0052 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 11
        else:
            rotation = -(0.0049 * deviation + 0.0005 * (deviation - self.deviation_left))
            speed = 9

        return speed, rotation

    def curve_case(self, deviation, dif):
        if abs(dif) < 50:
            rotation = -(0.01 * deviation + 0.0006 * (deviation - self.deviation_left))
        elif abs(dif) < 80:
            rotation = -(0.0092 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif abs(dif) < 130:
            rotation = -(0.0087 * deviation + 0.0005 * (deviation - self.deviation_left))
        elif abs(dif) < 190:
            rotation = -(0.008 * deviation + 0.0005 * (deviation - self.deviation_left))
        else:
            rotation = -(0.0075 * deviation + 0.0005 * (deviation - self.deviation_left))

        speed = 5
        return speed, rotation

    def get_point(self, index, img):
        mid = 0
        if np.count_nonzero(img[index]) > 0:
            left = np.min(np.nonzero(img[index]))
            right = np.max(np.nonzero(img[index]))
            mid = np.abs(left - right) / 2 + left
        return int(mid)

    def execute(self):
        image = self.camera.getImage().data
        if image.shape == (3, 3, 3):
            time.sleep(3)

        self.update_frame('frame_0', image)
        # cv2.imwrite(PRETRAINED_MODELS + 'montmelo_data/' + str(self.iteration) + '.jpg', image)
        # self.iteration += 1

        try:
            if 'NoiseType' in self.config:
                image = add_noise(image, self.config['NoiseParams'], 
                        noise_type = self.config['NoiseType'])
                self.update_frame('frame_1', image)

            image_cropped = image[230:, :, :]
            image_hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([180, 255, 255])
            image_mask = cv2.inRange(image_hsv, lower_red, upper_red)

            # show image in gui -> frame_0

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
                points.append((self.get_point(index, image_mask), index))

            points.append((self.get_point(rows, image_mask), rows))

            # We convert to show it
            # Shape gives us the number of rows and columns of an image
            size = image_mask.shape
            rows = size[0]
            columns = size[1]

            # We look for the position on the x axis of the pixels that have value 1 in different positions and
            position_x_down = np.where(image_mask[points[3][1], :])
            position_x_middle = np.where(image_mask[points[1][1], :])
            position_x_above = np.where(image_mask[points[2][1], :])

            # We see that white pixels have been located and we look if the center is located
            # In this way we can know if the car has left the circuit
            x_middle_left_down, not_found_down = self.check_center(position_x_down)
            x_middle_left_middle, not_found_middle = self.check_center(position_x_middle)

            # We look if white pixels of the row above are located
            if (len(position_x_above[0]) > 1):
                self.x_middle_left_above = (position_x_above[0][0] + position_x_above[0][
                    len(position_x_above[0]) - 1]) / 2
                # We look at the deviation from the central position. The center of the line is in position cols/2
                deviation = self.x_middle_left_above - (cols / 2)

                # If the row below has been lost we have a different case, which we treat as an exception
                if not_found_down == True:
                    speed, rotation = self.exception_case(x_middle_left_middle, deviation)
                else:
                    # We check is formula 1 is in curve or straight
                    dif = x_middle_left_down - self.x_middle_left_above
                    x = float(((-dif) * (310 - 350))) / float(260 - 350) + x_middle_left_down

                    if abs(x - x_middle_left_middle) < 2:
                        speed, rotation = self.straight_case(deviation, dif)
                    else:
                        speed, rotation = self.curve_case(deviation, dif)

                # We update the deviation
                self.deviation_left = deviation
            else:
                # If the formula 1 leaves the red line, the line is searched
                if self.x_middle_left_above > (columns / 2):
                    rotation = -1
                else:
                    rotation = 1
                speed = -0.6

            self.motors.sendV(speed)
            self.motors.sendW(rotation)
            # self.json_data.append({'v': speed, 'w': rotation})
            # with open(PRETRAINED_MODELS + 'montmelo_data/data.json', 'w') as outfile:
            # json.dump(self.json_data, outfile)

        except Exception as err:
            print(err)
