import rospy
from std_msgs.msg import Float32
import threading


def imuMsg2IMU(imuMsg):

    imu = IMU()

    imu.data = imuMsg.data
    now = rospy.get_rostime()
    imu.timeStamp = now.secs + (now.nsecs * 1e-9)

    return imu


class IMU():

    def __init__(self):

        self.data = 0  # X coord [meters]
        self.timeStamp = 0  # Time stamp [s]

    def __str__(self):
        s = "IMU: {\n   x: " + str(self.x) + "\n   timeStamp: " + str(self.timeStamp) + "\n}"
        return s


class ListenerIMU:
    '''
        ROS IMU Subscriber. IMU Client to Receive imu from ROS nodes.
    '''
    def __init__(self, topic):
        '''
        ListenerIMU Constructor.

        @param topic: ROS topic to subscribe
        @type topic: String

        '''
        self.topic = topic
        self.data = IMU()
        self.sub = None
        self.lock = threading.Lock()
        self.start()

    def __callback(self, imu):
        '''
        Callback function to receive and save IMU.

        @param odom: ROS Odometry received

        @type odom: Odometry

        '''
        imu = imuMsg2IMU(imu)

        self.lock.acquire()
        self.data = imu
        self.lock.release()

    def stop(self):
        '''
        Stops (Unregisters) the client.

        '''
        self.sub.unregister()

    def start(self):
        '''
        Starts (Subscribes) the client.

        '''
        self.sub = rospy.Subscriber(self.topic, Float32, self.__callback)

    def getIMU(self):
        '''
        Returns last IMU.

        @return last IMU saved

        '''
        self.lock.acquire()
        imu = self.data
        self.lock.release()

        return imu
