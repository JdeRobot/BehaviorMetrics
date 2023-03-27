import cv2
import numpy as np

from .utils import F1GazeboUtils


class F1GazeboSimplifiedPerception:
    def processed_image(self, img, height, width, x_row, center_image):
        """
        In FollowLine tasks, gets the centers of central line
        In Followlane Tasks, gets the center of lane

        :parameters: input image 640x480
        :return:
            centrals: lists with distance to center in pixels
            cntrals_normalized: lists with distance in range [0,1] for calculating rewards
        """
        image_middle_line = height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[x_row[idx], :] for idx, x in enumerate(x_row)]
        centrals = list(map(self.get_center, lines))

        centrals_normalized = [
            float(center_image - x) / (float(width) // 2)
            for _, x in enumerate(centrals)
        ]

        F1GazeboUtils.show_image_with_centrals(
           "centrals", mask, 5, centrals, centrals_normalized, x_row
        )

        return centrals, centrals_normalized


    @staticmethod
    def get_center(lines):
        ''' 
        takes center line and returns position regarding to it
        '''
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            return np.min(np.nonzero(lines)) + point
        except ValueError:
            return 0


    def calculate_observation(self, state, center_image, pixel_region: list) -> list:
        """
        returns list of states in range [-7,9] if self.num_regions = 16 => pixel_regions = 40
        state = -7 corresponds to center line far right
        state = 9 is far left
        """
        final_state = []
        for _, x in enumerate(state):
            final_state.append(int((center_image - x) / pixel_region) + 1)

        return final_state


    def calculate_centrals_lane(
        self, img, height, width, x_row, lower_limit, center_image
    ):
        image_middle_line = height // 2
        # cropped image from second half to bottom line
        img_sliced = img[image_middle_line:]
        # convert to black and white mask
        # lower_grey = np.array([30, 32, 22])
        # upper_grey = np.array([128, 128, 128])
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 110, 255, cv2.THRESH_BINARY)
        # get Lines to work for
        lines = [mask[x_row[idx], :] for idx, _ in enumerate(x_row)]
        # added last line (239), to control center line in bottom
        lines.append(mask[lower_limit, :])

        centrals_in_pixels = list(map(self.get_center_right_lane, lines))
        centrals_normalized = [
            abs(float(center_image - x) / (float(width) // 2))
            for _, x in enumerate(centrals_in_pixels)
        ]

        # F1GazeboUtils.show_image_with_centrals(
        #    "mask", mask, 5, centrals_in_pixels, centrals_normalized, self.x_row
        # )

        return centrals_in_pixels, centrals_normalized

    @staticmethod
    def get_center_right_lane(lines):
        try:
            # inversed line
            inversed_lane = [x for x in reversed(lines)]
            # cut off right blanks
            inv_index_right = np.argmin(inversed_lane)
            # cropped right blanks
            cropped_lane = inversed_lane[inv_index_right:]
            # cut off central line
            inv_index_left = np.argmax(cropped_lane)
            # get real lane index
            index_real_right = len(lines) - inv_index_right
            if inv_index_left == 0:
                index_real_left = 0
            else:
                index_real_left = len(lines) - inv_index_right - inv_index_left
            # get center lane
            center = (index_real_right - index_real_left) // 2
            center_lane = center + index_real_left

            # avoid finish line or other blank marks on the road
            if center_lane == 0:
                center_lane = 320

            return center_lane

        except ValueError:
            return 0

    @staticmethod
    def get_center_circuit_no_wall(lines):
        try:
            pos_final_linea_negra = np.argmin(lines) + 15
            carril_derecho_entero = lines[pos_final_linea_negra:]
            final_carril_derecho = np.argmin(carril_derecho_entero)
            lim_izq = pos_final_linea_negra
            lim_der = pos_final_linea_negra + final_carril_derecho

            punto_central_carril = (lim_der - lim_izq) // 2
            punto_central_absoluto = lim_izq + punto_central_carril
            return punto_central_absoluto

        except ValueError:
            return 0
