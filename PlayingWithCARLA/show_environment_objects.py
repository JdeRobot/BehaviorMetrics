import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time

print(carla.__file__)

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()
print(world)
time.sleep(2)


#print(world.get_environment_objects())
print(len(world.get_environment_objects()))

environment_objects = world.get_environment_objects()

print(type(environment_objects))
print(environment_objects[0].id)

environment_object_ids = []

for env_obj in environment_objects:
    environment_object_ids.append(env_obj.id)



environment_object_ids = set(environment_object_ids)
print(type(environment_object_ids))

world.enable_environment_objects(environment_object_ids, False)
#for env_obj in environment_objects:
    



