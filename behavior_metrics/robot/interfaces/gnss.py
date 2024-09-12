import rospy
from sensor_msgs.msg import NavSatFix
import threading


def gnssMsg2GNSS(gnssMsg):

    gnss = GNSS()

    #print(gnssMsg)

    gnss.latitude = gnssMsg.latitude
    gnss.longitude = gnssMsg.longitude
    gnss.altitude = gnssMsg.altitude

    #msg.latitude = data[0]
    #    msg.longitude = data[1]
    #    msg.altitude = data[2]
    
    now = rospy.get_rostime()
    gnss.timeStamp = now.secs + (now.nsecs * 1e-9)

    return gnss


class GNSS():

    def __init__(self):

        self.latitude = 0
        self.longitude = 0
        self.altitude = 0
        self.timeStamp = 0  # Time stamp [s]

    def __str__(self):
        s = "GNSS: {\n   latitude: " + str(self.latitude) + "\n }\n  longitude: " + str(self.longitude) + "\n }\n  altitude: " + str(self.altitude) + "\n }\n  timeStamp: " + str(self.timeStamp) + "\n}"
        return s


class ListenerGNSS:
    '''
        ROS GNSS Subscriber. GNSS Client to Receive gnss from ROS nodes.
    '''
    def __init__(self, topic):
        '''
        ListenerGNSS Constructor.

        @param topic: ROS topic to subscribe
        @type topic: String

        '''
        self.topic = topic
        self.data = GNSS()
        self.sub = None
        self.lock = threading.Lock()
        self.start()

    def __callback(self, gnss):
        '''
        Callback function to receive and save GNSS.

        @param odom: ROS Odometry received

        @type odom: Odometry

        '''
        gnss = gnssMsg2GNSS(gnss)

        self.lock.acquire()
        self.data = gnss
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
        self.sub = rospy.Subscriber(self.topic, NavSatFix, self.__callback)

    def getGNSS(self):
        '''
        Returns last GNSS.

        @return last GNSS saved

        '''
        self.lock.acquire()
        gnss = self.data
        self.lock.release()

        return gnss
