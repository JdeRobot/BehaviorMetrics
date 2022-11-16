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



camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car)
image_queue = queue.Queue()
camera.listen(image_queue.put)

#rgb camera
image = image_queue.get()
image = image_queue.get()
cc = carla.ColorConverter.Raw
image.save_to_disk('_out/1.png', cc)


img = cv2.imread("_out/1.png", cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(car.get_velocity())
plt.imshow(img)
plt.show()


