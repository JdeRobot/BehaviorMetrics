
import carla
import numpy as np
from carla.agents.navigation.basic_agent import BasicAgent


try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
    import queue
except ImportError:
    import Queue as queue



def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def main():
    # all the actors in the world. For destroying later.
    actor_list = []
    pygame.init()

    # create client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # access world from client
    world = client.get_world()

    # Enable synchronous mode
    print('Enabling synchronous mode')
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    try:
        # set weather conditions
        world.set_weather(carla.WeatherParameters.ClearNoon)

        # define location
        map = world.get_map()
        spawn_points = map.get_spawn_points()
        hero_transform = spawn_points[97]

        # Get the van to spawn in front of the hero
        # Get the waypoint of the hero, since the spawn points are only Transforms
        hero_waypoint = map.get_waypoint(hero_transform.location)

        # Get the waypoint 15 meters in front of it
        van_waypoint = hero_waypoint.next(15.0)
        van_transform = van_waypoint[0].transform

        # spawn higher or it will get stuck
        van_transform.location.z += 0.5

        # get all the blueprints in this world
        blueprint_library = world.get_blueprint_library()
        # define the blueprint of hero vehicle
        prius_bp = blueprint_library.find('vehicle.toyota.prius')
        white = '255,255,255'
        prius_bp.set_attribute('color', white)

        # blueprint colavan
        colavan_bp = blueprint_library.find('vehicle.carlamotors.carlacola')

        # spawn our hero
        hero = world.spawn_actor(prius_bp, hero_transform)
        # add actor to the list for destruction, otherwise vehicle is stuck in there forever
        actor_list.append(hero)

        # spawn van
        colavan = world.spawn_actor(colavan_bp, van_transform)
        actor_list.append(colavan)

        # add a camera
        camera_class = blueprint_library.find('sensor.camera.rgb')
        camera_class.set_attribute('image_size_x', '600')
        camera_class.set_attribute('image_size_y', '600')
        camera_class.set_attribute('fov', '90')
        camera_class.set_attribute('sensor_tick', '0.1')
        cam_transform1 = carla.Transform(carla.Location(x=1.8, z=1.3))
        # cam_transform2 = cam_transform1 + carla.Location(y=0.54)

        # # spawn camera to hero
        camera1 = world.spawn_actor(camera_class, cam_transform1, attach_to=hero)
        actor_list.append(camera1)
        # camera2 = world.spawn_actor(camera_class, cam_transform2, attach_to=hero)
        # actor_list.append(camera2)

        # Makes a sync queue for the sensor data
        image_queue1 = queue.Queue()
        camera1.listen(image_queue1.put)
        frame = None

        # image_queue2 = queue.Queue()
        # camera2.listen(image_queue2.put)

        # Note its going to drive at consistent speed 15 km/h
        # PID controller gets unstable after 15 km/h
        #roaming_prius = BasicAgent(hero, target_speed=15)
        #destiny = spawn_points[96].location
        #roaming_prius.set_destination((destiny.x, destiny.y, destiny.z))

        roaming_van = BasicAgent(colavan, target_speed=15)
        #roaming_van.set_destination((destiny.x, destiny.y, destiny.z))

        # If you want the hero to drive around in autopilot
        hero.set_autopilot(True)

        display = pygame.display.set_mode((600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)

        # tracks time and frame rate management class.
        clock = pygame.time.Clock()

        while True:
            clock.tick()
            world.tick()
            ts = world.tick()

            # Get control commands
            #control_hero = roaming_prius.run_step()
            #hero.apply_control(control_hero)

            #control_van = roaming_van.run_step()
            #colavan.apply_control(control_van)

            if frame is not None:
                if ts != frame + 1:
                    print('frame skip!')
                    print("frame skip!")
            print(ts)
            frame = ts

            while True:
                image1 = image_queue1.get()
                print(image1)
                # as long as the image number == frame count you are fine and this loop is not necessary
                print("image1.frame_number: {} % ts: {}".format(image1.frame_number, ts))
                if image1.frame_number == ts:
                    break
                print(
                    'wrong image time-stampstamp: frame=%d, image.frame=%d',
                    ts,
                    image1.frame_number)

            draw_image(display, image1)

            # reset display
            pygame.display.flip()

    finally:
        print('Disabling synchronous mode')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print("pygame quit, done")

main()