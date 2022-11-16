from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=150, height=300),
    pixels_per_meter=10,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

birdview_producer = BirdViewProducer(
    client,  # carla.Client
    target_size=PixelDimensions(width=100, height=300),
    pixels_per_meter=10,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

world = client.get_world()
print(world.get_actors())
car = world.get_actors().filter('vehicle.*')[0]
# Input for your model - call it every simulation step
# returned result is np.ndarray with ones and zeros of shape (8, height, width)
birdview = birdview_producer.produce(
    agent_vehicle=car  # carla.Actor (spawned vehicle)
)

import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

while True:
    # Use only if you want to visualize
    # produces np.ndarray of shape (height, width, 3)
    rgb = BirdViewProducer.as_rgb(birdview)
    #print(rgb)
    '''
    for x, im in enumerate(rgb):
        for y, im_2 in enumerate(rgb[x]):
            #print(rgb[x][y])
            #print(np.array([0,0,0]))
            #print((rgb[x][y] == np.array([0,0,0])).all())
            if (rgb[x][y] == np.array([0,0,0])).all() != True:
                print(rgb[x][y])

    '''
    plt.imshow(rgb)
    plt.show()

    print(rgb.shape)

    #time.sleep(1)
