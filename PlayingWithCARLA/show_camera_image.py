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

try:
    while True:
        image = image_queue.get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # RGBA format
        image = array[:, :, :3] #  Take only RGB
        image = image[:, :, ::-1] # BGR
        image = image[:, ::-1, :] # Mirror

        image = image[325:600, 0:800]
        #plt.imshow(image)
        #plt.show()

        img = cv2.resize(image, (200, 66))

        # Clear screen to white before drawing 
        screen.fill((255, 255, 255))
        
        # Convert to a surface and splat onto screen offset by border width and height
        surface = pygame.surfarray.make_surface(image)
        screen.blit(surface, (0, 0))
        screen.blit(pygame.transform.rotate(screen, 270), (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        

except KeyboardInterrupt:
    pass

