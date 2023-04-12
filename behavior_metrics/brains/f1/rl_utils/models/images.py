import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image as im
import rospy
from sensor_msgs.msg import Image as ImageROS
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from ..image_f1 import Image, ListenerCamera, Image
from .utils import F1GazeboUtils


class F1GazeboImages:
    def __init__(self):
        self.f1gazeboutils = F1GazeboUtils()
        self.image = Image()

    def get_camera_info(self):
        image_data = None
        f1_image_camera = None
        success = False

        while image_data is None or success is False:
            image_data = rospy.wait_for_message(
                "/F1ROS/cameraL/image_raw", ImageROS, timeout=5
            )
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            # f1_image_camera = image_msg_to_image(image_data, cv_image)
            if np.any(cv_image):
                success = True

        return f1_image_camera, cv_image

    def image_msg_to_image(self, img, cv_image):
        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image

    def image_preprocessing_black_white_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_White_3D = np.expand_dims(mask, axis=2)

        return mask_black_White_3D

    def image_preprocessing_black_white_32x32(self, img, height):
        image_middle_line = height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 120, 120), (0, 255, 255))
        _, mask = cv2.threshold(line_pre_proc, 48, 255, cv2.THRESH_BINARY)
        mask_black_white_32x32 = cv2.resize(mask, (32, 32), cv2.INTER_AREA)
        mask_black_white_32x32 = np.expand_dims(mask_black_white_32x32, axis=2)

        self.f1gazeboutils.show_image("mask32x32", mask_black_white_32x32, 5)
        return mask_black_white_32x32

    def image_preprocessing_gray_32x32(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        img_gray_3D = cv2.resize(img_proc, (32, 32), cv2.INTER_AREA)
        img_gray_3D = np.expand_dims(img_gray_3D, axis=2)

        return img_gray_3D

    def image_preprocessing_raw_original_size(self, img):
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        return img_sliced

    def image_preprocessing_color_quantization_original_size(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=50)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)

        return kmeans.cluster_centers_[labels].reshape(w, h, -1)

    def image_preprocessing_color_quantization_32x32x1(self, img):
        n_colors = 3
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        img_sliced = np.array(img_sliced, dtype=np.float64) / 255
        w, h, d = original_shape = tuple(img_sliced.shape)
        image_array = np.reshape(img_sliced, (w * h, d))
        image_array_sample = shuffle(image_array, random_state=0, n_samples=500)
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
        labels = kmeans.predict(image_array)
        im_reshape = kmeans.cluster_centers_[labels].reshape(w, h, -1)
        im_resize32x32x1 = np.expand_dims(np.resize(im_reshape, (32, 32)), axis=2)

        return im_resize32x32x1

    def image_preprocessing_reducing_color_PIL_original_size(self, img):
        num_colors = 4
        image_middle_line = self.height // 2
        img_sliced = img[image_middle_line:]

        array2pil = im.fromarray(img_sliced)
        array2pil_reduced = array2pil.convert(
            "P", palette=im.ADAPTIVE, colors=num_colors
        )
        pil2array = np.expand_dims(np.array(array2pil_reduced), 2)
        return pil2array

    def image_callback(self, image_data):
        self.image_raw_from_topic = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

    def processed_image_circuit_no_wall(self, img):
        """
        detecting middle of the right lane
        """
        image_middle_line = self.height // 2
        # cropped image from second half to bottom line
        img_sliced = img[image_middle_line:]
        # convert to black and white mask
        # lower_grey = np.array([30, 32, 22])
        # upper_grey = np.array([128, 128, 128])
        img_gray = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        # mask = cv2.inRange(img_sliced, lower_grey, upper_grey)

        # img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        # line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))
        # _, : list[ndarray[Any, dtype[generic]]]: list[ndarray[Any, dtype[generic]]]: list[ndarray[Any, dtype[generic]]]mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[self.x_row[idx], :] for idx, x in enumerate(self.x_row)]
        # added last line (239) only FOllowLane, to break center line in bottom
        # lines.append(mask[self.lower_limit, :])

        centrals_in_pixels = list(map(self.get_center, lines))
        centrals_normalized = [
            float(self.center_image - x) / (float(self.width) // 2)
            for _, x in enumerate(centrals_in_pixels)
        ]

        self.show_image("mask", mask, 5)

        return centrals_in_pixels, centrals_normalized
