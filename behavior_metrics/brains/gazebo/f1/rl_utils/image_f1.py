import threading
import rospy

import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image as ImageROS


def imageMsg2Image(img, bridge):
    image = Image()
    image.width = img.width
    image.height = img.height
    image.format = "RGB8"
    image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
    cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    image.data = cv_image
    return image


class Image:

    def __init__(self):
        self.height = 3  # Image height [pixels]
        self.width = 3  # Image width [pixels]
        self.timeStamp = 0  # Time stamp [s] */
        self.format = ""  # Image format string (RGB8, BGR,...)
        self.data = np.zeros((self.height, self.width, 3), np.uint8)  # The image data itself
        self.data.shape = self.height, self.width, 3

    def __str__(self):
        s = "Image: {\n   height: " + str(self.height) + "\n   width: " + str(self.width)
        s = s + "\n   format: " + self.format + "\n   timeStamp: " + str(self.timeStamp)
        s = s + "\n   data: " + str(self.data) + "\n}"
        return s


class ListenerCamera:

    def __init__(self, topic):
        self.topic = topic
        self.data = Image()
        self.sub = None
        self.lock = threading.Lock()
        self.total_frames = 0

        self.bridge = CvBridge()
        self.start()
        

    def __callback(self, img):
        self.total_frames += 1
        image = imageMsg2Image(img, self.bridge)

        self.lock.acquire()
        self.data = image
        self.lock.release()

    def stop(self):
        self.sub.unregister()

    def start(self):
        self.sub = rospy.Subscriber(self.topic, ImageROS, self.__callback)

    def getImage(self):
        self.lock.acquire()
        image = self.data
        self.lock.release()

        return image

    def getTopic(self):
        return self.topic

    def hasproxy(self):
        return hasattr(self, "sub") and self.sub