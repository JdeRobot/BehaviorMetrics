import rospy
from geometry_msgs.msg import Twist
import threading
from .threadPublisher import ThreadPublisher

try:
    from carla_msgs.msg import CarlaEgoVehicleControl
except ModuleNotFoundError as ex:
    print('CARLA is not supported')


def cmdvel2Twist(vel):

    tw = Twist()
    tw.linear.x = vel.vx
    tw.linear.y = vel.vy
    tw.linear.z = vel.vz
    tw.angular.x = vel.ax
    tw.angular.y = vel.ay
    tw.angular.z = vel.az

    return tw


def cmdvel2CarlaEgoVehicleControl(vel):
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

    def __init__(self, topic, maxV, maxW, v, w):

        self.maxW = maxW
        self.maxV = maxV
        self.v = v
        self.w = w
        self.topic = topic
        self.data = CMDVel()
        self.pub = rospy.Publisher(self.topic, Twist, queue_size=1)
        rospy.init_node("FollowLineF1")
        self.lock = threading.Lock()
        self.kill_event = threading.Event()
        self.thread = ThreadPublisher(self, self.kill_event)
        self.thread.daemon = True
        self.start()

    def publish(self):
        self.lock.acquire()
        tw = cmdvel2Twist(self.data)
        self.lock.release()
        self.pub.publish(tw)

    def stop(self):
        self.kill_event.set()
        self.pub.unregister()

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

    def __init__(self, topic, maxV, maxW, v, w):

        self.maxW = maxW
        self.maxV = maxV
        self.v = v
        self.w = w
        self.topic = topic
        self.data = CARLAVel()
        self.pub = rospy.Publisher(self.topic, CarlaEgoVehicleControl, queue_size=1)
        rospy.init_node("CARLAMotors")
        self.lock = threading.Lock()
        self.kill_event = threading.Event()
        self.thread = ThreadPublisher(self, self.kill_event)
        self.thread.daemon = True
        self.start()

    def publish(self):
        self.lock.acquire()
        vehicle_control = cmdvel2CarlaEgoVehicleControl(self.data)
        self.lock.release()
        self.pub.publish(vehicle_control)

    def stop(self):
        self.kill_event.set()
        self.pub.unregister()

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
