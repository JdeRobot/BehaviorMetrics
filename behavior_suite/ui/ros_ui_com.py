import rospy
from std_msgs.msg import String

class Communicator:

    def __init__(self):
        self.pub = rospy.Publisher('/behavior/ui_comm', String, queue_size=10)

    def send_msg(self, key):
        self.pub.publish(str(key))
    
    def send_topics(self, topics):
        pass

