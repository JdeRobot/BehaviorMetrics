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


car.set_autopilot(True)

while True:
    print(car.get_vehicle_control())
