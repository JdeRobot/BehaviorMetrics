import carla
import queue
import matplotlib.pyplot as plt
import cv2
import time
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
import queue
import pygame


h,w=800,800
h,w=200, 200

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
world = client.get_world()

time.sleep(2)

world.get_actors()
car = world.get_actors().filter('vehicle.*')[0]

camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car)
image_queue = queue.Queue()
camera.listen(image_queue.put)

#rgb camera
image = image_queue.get()

pygame.init()
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("Serious Work - not games")
done = False
clock = pygame.time.Clock()

# Get a font for rendering the frame number
basicfont = pygame.font.SysFont(None, 32)

array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
image = array[:, :, :3] #  Take only RGB
image = image[:, :, ::-1] # BGR

import tensorflow as tf
import numpy as np
from gradcam import GradCAM


from albumentations import (
    Compose, Normalize
)

# LOAD TF MODEL
PRETRAINED_MODELS = "/home/jderobot/Documents/Projects/"
model = "20220916-164609_pilotnet_CARLA_extreme_cases_16_09_dataset_new_crop_extreme_cases.h5"
net = tf.keras.models.load_model(PRETRAINED_MODELS + model)
print(net.summary())

try:
    while True:
        image = image_queue.get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
        image = array[:, :, :3] #  Take only RGB
        image = image[:, :, ::-1] # BGR
        image = image[:, ::-1, :] # Mirror

        original_image = image[325:600, 0:800]
        #plt.imshow(image)
        #plt.show()

        resized_image = cv2.resize(original_image, (200, 66))
        AUGMENTATIONS_TEST = Compose([
            Normalize()
        ])
        image = AUGMENTATIONS_TEST(image=resized_image)
        img = image["image"]


        img = np.expand_dims(img, axis=0)
        prediction = net.predict(img)

        prediction_w = prediction[0][1] * (1 - (-1)) + (-1)
        throttle = prediction[0][0]
        steer = prediction_w
        print(float(throttle), steer)

        #car.apply_control(carla.VehicleControl(throttle=float(throttle), steer=steer))
        car.apply_control(carla.VehicleControl(throttle=0.2, steer=steer))

        i = np.argmax(prediction[0])
        cam = GradCAM(net, i)
        heatmap = cam.compute_heatmap(img)
        heatmap = cv2.resize(heatmap, (heatmap.shape[1], heatmap.shape[0]))
        
        print(original_image.shape)
        print(resized_image.shape)
        print(heatmap.shape)
        (heatmap, output) = cam.overlay_heatmap(heatmap, resized_image, alpha=0.5)
        print(output.shape)

        # Clear screen to white before drawing 
        screen.fill((255, 255, 255))
        
        # Convert to a surface and splat onto screen offset by border width and height
        #surface = pygame.surfarray.make_surface(original_image)
        surface = pygame.surfarray.make_surface(output)
        screen.blit(surface, (0, 0))
        screen.blit(pygame.transform.rotate(screen, 270), (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        

except KeyboardInterrupt:
    pass

