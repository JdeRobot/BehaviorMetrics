import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import tensorflow as tf
import numpy as np

from albumentations import (
    Compose, Normalize
)


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



# LOAD CAMERA
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car)
image_queue = queue.Queue()
camera.listen(image_queue.put)
#rgb camera
image = image_queue.get()

# LOAD TF MODEL
PRETRAINED_MODELS = "/home/jderobot/Documents/Projects/"
model = "20220907-140021_pilotnet_CARLA_extreme_cases_07_09_cp.h5"

print('***********************************************************************************************')
net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
#net = tf.saved_model.load(PRETRAINED_MODELS + model)
print('***********************************************************************************************')


car.set_autopilot(True)

while True:
    # Predict
    image = image_queue.get()

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
    image = array[:, :, :3] #  Take only RGB
    image = image[:, :, ::-1] # BGR

    image = image[300:600, 0:800]

    img = cv2.resize(image, (200, 66))
    AUGMENTATIONS_TEST = Compose([
        Normalize()
    ])
    image = AUGMENTATIONS_TEST(image=img)
    img = image["image"]


    img = np.expand_dims(img, axis=0)
    prediction = net.predict(img)
    prediction_w = prediction[0][1] * (1 - (-1)) + (-1)

    throttle = prediction[0][0]
    steer = prediction_w
    
    print(abs(steer)-abs(car.get_vehicle_control().steer))

    #print(float(throttle), steer)
    #print(car.get_vehicle_control())
    
    
    
    
    
