import rospy
from sensor_msgs.msg import Imu
import threading


def imuMsg2IMU(imuMsg):

    imu = IMU()

    imu.compass = imuMsg.orientation
    imu.accelerometer = imuMsg.linear_acceleration
    imu.gyroscope = imuMsg.angular_velocity
    
    now = rospy.get_rostime()
    imu.timeStamp = now.secs + (now.nsecs * 1e-9)

    return imu


class IMU():

    def __init__(self):

        self.compass = {
            'x': 0,
            'y': 0,
            'z': 0,
            'w': 0
        } 
        self.gyroscope = {
            'x': 0,
            'y': 0,
            'z': 0
        } 
        self.accelerometer = {
            'x': 0,
            'y': 0,
            'z': 0
        } 
        self.timeStamp = 0  # Time stamp [s]

    def __str__(self):
        s = "IMU: {\n   compass: " + str(self.compass) + "\n }\n  accelerometer: " + str(self.accelerometer) + "\n }\n  gyroscope: " + str(self.gyroscope) + "\n }\n  timeStamp: " + str(self.timeStamp) + "\n}"
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
        self.sub = rospy.Subscriber(self.topic, Imu, self.__callback)

    def getIMU(self):
        '''
        Returns last IMU.

        @return last IMU saved

        '''
        self.lock.acquire()
        imu = self.data
        self.lock.release()

        return imu
