# Modified based on: https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/generate_traffic.py
import carla
from utils.logger import logger
import random

spawn_actor = carla.command.SpawnActor
set_autopilot = carla.command.SetAutopilot
future_actor = carla.command.FutureActor

class TrafficManager:
    def __init__(self, n_vehicle, n_walker, percentage_walker_running=0.0, percentage_walker_crossing=0.0, async_mode=False, port=8000):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        if not async_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True 
            self.world.apply_settings(settings)

        traffic_manager = self.client.get_trafficmanager(port)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        self.traffic_manager = traffic_manager

        self.n_vehicle = n_vehicle
        self.n_walker = n_walker
        self.percentage_walker_running = percentage_walker_running
        self.percentage_walker_crossing = percentage_walker_crossing

        self.vehicles = []
        self.walkers = []
        self.walker_actors = []
        self.walker_ids = []
    
    def generate_traffic(self):
        for p in self.world.get_actors():
            if 'vehicle' in p.type_id:
                if p.attributes['role_name'] != 'ego_vehicle':
                    p.set_autopilot(True, self.traffic_manager.get_port())
                    
        if self.n_vehicle > 0:
            self.spawn_vehicles(self.world, self.client, self.n_vehicle, self.traffic_manager)
        if self.n_walker > 0:
            self.spawn_pedestrians(self.world, self.client, self.n_walker, 
                                percentagePedestriansRunning=self.percentage_walker_running, 
                                percentagePedestriansCrossing=self.percentage_walker_crossing)
        logger.info('spawned %d vehicles and %d walkers.' % (len(self.vehicles), len(self.walkers)))

    def get_actor_blueprints(self, world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return bps

        # If the filter returns only one bp, we assume that this one needed and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []


    def get_pedestrain_spawn_points(self, world, n):
        spawn_points = []
        for i in range(n):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        return spawn_points


    def get_vehicle_spawn_points(self, world, n_vehicles):
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if n_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logger.warning(msg, n_vehicles, number_of_spawn_points)
            n_vehicles = number_of_spawn_points
        return spawn_points

    def spawn_vehicles(self, world, client, n_vehicles, traffic_manager):
        blueprints = self.get_actor_blueprints(world, 'vehicle.*', 'All')
        # blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car'] # cars only
        blueprints = sorted(blueprints, key=lambda bp: bp.id)
        spawn_points = self.get_vehicle_spawn_points(world, n_vehicles)

        vehicles_list = []
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')


            batch.append(spawn_actor(blueprint, transform)
                .then(set_autopilot(future_actor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, True):
            if response.error:
                logger.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
       
        self.vehicles = vehicles_list

    def spawn_pedestrians(self, world, client, n_pedestrians, percentagePedestriansRunning=0.0, percentagePedestriansCrossing=0.0):
        walkers_list = []
        all_id = []

        # 1. get spawn points and blueprints
        spawn_points = self.get_pedestrain_spawn_points(world, n_pedestrians)
        blueprintsWalkers = self.get_actor_blueprints(world, 'walker.pedestrian.*', '2')

        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(spawn_actor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logger.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(spawn_actor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logger.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        
        self.walkers = walkers_list
        self.walker_actors = all_actors
        self.walker_ids = all_id
    
    def destroy(self):
        logger.info('destroying %d vehicles' % len(self.vehicles))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles])

        for i in range(0, len(self.walker_ids), 2):
            self.walker_actors[i].stop()

        logger.info('destroying %d walkers' % len(self.walkers))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_ids])
