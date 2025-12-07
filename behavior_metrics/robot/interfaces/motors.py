# import rospy
import os
from utils.logger import logger

from .threadPublisher import ThreadPublisher

ROS_VERSION = os.environ.get('ROS_VERSION  ', "None")
USE_ROS = ROS_VERSION   in ('1', '2')


if USE_ROS:
    if ROS_VERSION == '2':
        import rclpy
        from rclpy.node import Node
        from  geometry_msgs.msg import Twist
        from carla_msgs.msg import CarlaEgoVehicleControl
    else:
        import rospy
        from geometry_msgs.msg import Twist
        from carla_msgs.msg import CarlaEgoVehicleControl
else:
    #Python API
    Node = None
    Twist = None
    CarlaEgoVehicleControl = None



def cmdvel2Twist(vel):
    if Twist is None:
        return None

    tw = Twist()
    tw.linear.x = float(vel.vx)  
    tw.linear.y =  float(vel.vy)
    tw.linear.z = float(vel.vz)
    tw.angular.x = float(vel.ax)
    tw.angular.y = float(vel.ay)
    tw.angular.z = float(vel.az)

    return tw


def cmdvel2CarlaEgoVehicleControl(vel):
    if CarlaEgoVehicleControl is None:
        return None
    
    vehicle_control = CarlaEgoVehicleControl()
    vehicle_control.throttle = vel.throttle
    vehicle_control.steer = vel.steer
    vehicle_control.brake = vel.brake
    vehicle_control.hand_brake = False
    vehicle_control.reverse = False
    vehicle_control.gear = 0
    vehicle_control.manual_gear_shift = False

    return vehicle_control



class CMDVel():

    def __init__(self):

        self.vx = 0  # vel in x[m/s] (use this for V in wheeled robots)
        self.vy = 0  # vel in y[m/s]
        self.vz = 0  # vel in z[m/s]
        self.ax = 0  # angular vel in X axis [rad/s]
        self.ay = 0  # angular vel in X axis [rad/s]
        self.az = 0  # angular vel in Z axis [rad/s] (use this for W in wheeled robots)
        self.timeStamp = 0  # Time stamp [s]
        self.v = 0  # vel[m/s]
        self.w = 0  # angular vel [rad/s]

    def __str__(self):
        s = "CMDVel: {\n   vx: " + str(self.vx) + "\n   vy: " + str(self.vy)
        s = s + "\n   vz: " + str(self.vz) + "\n   ax: " + str(self.ax)
        s = s + "\n   ay: " + str(self.ay) + "\n   az: " + str(self.az)
        s = s + "\n   timeStamp: " + str(self.timeStamp) + "\n}"
        return s

class CARLAVel():

    def __init__(self):

        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.hand_brake = False
        self.reverse = False
        self.gear = 0
        self.manual_gear_shift = False

    def __str__(self):
        s = "CARLAVel: {\n   throttle: " + str(self.throttle) + "\n   steer: " + str(self.steer)
        s = s + "\n   brake: " + str(self.brake) + "\n   hand_brake: " + str(self.hand_brake)
        s = s + "\n   reverse: " + str(self.reverse) + "\n   gear: " + str(self.gear)
        s = s + "\n   manual_gear_shift: " + str(self.manual_gear_shift) + "\n}"
        return s

class PublisherMotors:

    def __init__(self, node: None, topic: str, maxV, maxW, v, w):
        self.node = node
        self.maxW = maxW
        self.maxV = maxV
        self.v = v
        self.w = w
        self.topic = topic
        self.data = CMDVel()
        
        if USE_ROS:
            if ROS_VERSION  == '2':
                self.pub = self.node.create_publisher(Twist, self.topic, 1)
            else:   
                self.pub = rospy.Publisher(self.topic, Twist, queue_size=1)
                rospy.init_node("FollowLineF1")
        else:
            # Python API
            self.pub = None
            
        self.lock = threading.Lock()
        self.kill_event = threading.Event()
        self.thread = ThreadPublisher(self, self.kill_event)
        self.thread.daemon = True
        self.start()

    def publish(self):
        if USE_ROS and self.pub is not None:
            with self.lock:
                msg = cmdvel2Twist(self.data)
            if msg:
                self.pub.publish(msg)
       
        # self.lock.acquire()
        # tw = cmdvel2Twist(self.data)
        # self.lock.release()
        # self.pub.publish(tw)

    def stop(self):
        self.kill_event.set()
        if USE_ROS and self.pub is not None:
            try:
                if ROS_VERSION == '2':
                    self.node.destroy_publisher(self.pub)
                else:
                    self.pub.unregister()
            except Exception as e:
                logger.warning(f"Error stopping publisher: {e}")
            self.pub = None

    def start(self):

        self.kill_event.clear()
        self.thread.start()

    def getTopic(self):
        return self.topic

    def getMaxW(self):
        return self.maxW

    def getMaxV(self):
        return self.maxV

    def sendVelocities(self, vel):

        self.lock.acquire()
        self.data = vel
        self.lock.release()

    def sendV(self, v):

        self.sendVX(v)
        self.v = v

    def sendL(self, l):

        self.sendVY(l)

    def sendW(self, w):

        self.sendAZ(w)
        self.w = w

    def sendVX(self, vx):

        self.lock.acquire()
        self.data.vx = vx
        self.lock.release()

    def sendVY(self, vy):

        self.lock.acquire()
        self.data.vy = vy
        self.lock.release()

    def sendAZ(self, az):

        self.lock.acquire()
        self.data.az = az
        self.lock.release()


class PublisherCARLAMotors:

    def __init__(self, node: None, topic: str, maxV, maxW, v, w):
        self.node = node
        self.maxW = maxW
        self.maxV = maxV
        self.v = v
        self.w = w
        self.topic = topic
        self.data = CARLAVel()
        
        if USE_ROS:
            if ROS_VERSION  == '2':
                self.pub = self.node.create_publisher(CarlaEgoVehicleControl, self.topic, 1)
            else:  
                self.pub = rospy.Publisher(self.topic, CarlaEgoVehicleControl, queue_size=1)
                rospy.init_node("CARLAMotors")
        else:
            # Python API
            self.pub = None
            
        self.lock = threading.Lock()
        self.kill_event = threading.Event()
        self.thread = ThreadPublisher(self, self.kill_event)
        self.thread.daemon = True
        self.start()

    def publish(self):
        if USE_ROS and self.pub is not None:
            with self.lock:
                msg = cmdvel2CarlaEgoVehicleControl(self.data)
            if msg:
                self.pub.publish(msg)

    def stop(self):
        self.kill_event.set()
        if USE_ROS and self.pub is not None:
            try:
                if ROS_VERSION == '2':
                    self.node.destroy_publisher(self.pub)
                else:
                    self.pub.unregister()
            except Exception as e:
                logger.warning(f"Error stopping publisher: {e}")
            self.pub = None
      
    def start(self):

        self.kill_event.clear()
        self.thread.start()

    def getTopic(self):
        return self.topic

    def getMaxW(self):
        return self.maxW

    def getMaxV(self):
        return self.maxV

    def sendVelocities(self, vel):

        self.lock.acquire()
        self.data = vel
        self.lock.release()

    def sendThrottle(self, throttle):

        self.lock.acquire()
        self.data.throttle = throttle
        self.lock.release()
        self.throttle = throttle

    def sendSteer(self, steer):

        self.lock.acquire()
        self.data.steer = steer
        self.lock.release()
        self.steer = steer

    def sendBrake(self, brake):

        self.lock.acquire()
        self.data.brake = brake
        self.lock.release()
        self.brake = brake
