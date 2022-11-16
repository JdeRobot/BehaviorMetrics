from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import csv
from os import listdir
from os.path import isfile, join
import pandas

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()

blueprint = world.get_blueprint_library().filter('vehicle')[0]
print(blueprint)
spawn_point = world.get_map().get_spawn_points()[0]
print(spawn_point)
player = world.try_spawn_actor(blueprint, spawn_point)