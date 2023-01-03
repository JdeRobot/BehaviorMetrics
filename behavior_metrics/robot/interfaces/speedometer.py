import rospy
from std_msgs.msg import Float32
import threading


def speedometer2Speedometer(speedometer):

    speed = Speedometer()

    speed.data = speedometer.data
    now = rospy.get_rostime()
    speed.timeStamp = now.secs + (now.nsecs * 1e-9)

    return speed


class Speedometer():

    def __init__(self):

        self.data = 0  # X coord [meters]
        self.timeStamp = 0  # Time stamp [s]

    def __str__(self):
        s = "Speedometer: {\n   x: " + str(self.x) + "\n   timeStamp: " + str(self.timeStamp) + "\n}"
        return s


class ListenerSpeedometer:
    '''
        ROS Speedometer Subscriber. Speedometer Client to Receive speedometer from ROS nodes.
    '''
    def __init__(self, topic):
        '''
        ListenerSpeedometer Constructor.

        @param topic: ROS topic to subscribe
        @type topic: String

        '''
        self.topic = topic
        self.data = Speedometer()
        self.sub = None
        self.lock = threading.Lock()
        self.start()

    def __callback(self, speedometer):
        '''
        Callback function to receive and save Speedometer.

        @param odom: ROS Odometry received

        @type odom: Odometry

        '''
        speedometer = speedometer2Speedometer(speedometer)

        self.lock.acquire()
        self.data = speedometer
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

    def getSpeedometer(self):
        '''
        Returns last Speedometer.

        @return last Speedometer saved

        '''
        self.lock.acquire()
        speedometer = self.data
        self.lock.release()

        return speedometer
