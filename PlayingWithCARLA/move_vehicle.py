import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time

client = carla.Client('localhost', 2000)

client.set_timeout(10.0) # seconds
world = client.get_world()
print(world)

time.sleep(2)

world.get_actors()

car = world.get_actors().filter('vehicle.*')[0]
print(car)
print(car.get_velocity())
print(car.get_vehicle_control())
#vehicle_control =car.get_vehicle_control()
#vehicle_control.steer = 10000


car.apply_control(carla.VehicleControl(throttle = 1, brake = 0))
print(car.get_velocity())
print(car.get_vehicle_control())
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
car.apply_control(carla.VehicleControl(throttle = 0, brake = 0))
time.sleep(5)
print(car.get_velocity())
print(car.get_vehicle_control())
'''
vehicle_control = carla.VehicleControl()
vehicle_control.steer = 10000

vehicle_control.manual_gear_shift =True
car.apply_control(vehicle_control)
time.sleep(0.5)
print(car.get_velocity())
car.apply_control(vehicle_control)
time.sleep(0.5)
print(car.get_velocity())
car.apply_control(vehicle_control)
time.sleep(0.5)
print(car.get_velocity())
car.apply_control(vehicle_control)
time.sleep(0.5)
print(car.get_velocity())
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)
car.apply_control(vehicle_control)
time.sleep(0.5)

print(car.get_velocity())
'''
