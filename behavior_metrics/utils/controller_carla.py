#!/usr/bin/env python

"""This module contains the controller of the application.

This application is based on a type of software architecture called Model View Controller. This is the controlling part
of this architecture (controller), which communicates the logical part (model) with the user interface (view).

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import shlex
import subprocess
import threading
import weakref  #
import cv2

# import rospy
import os
import time

# import rosbag
import json
import math
from utils.logger import logger

from datetime import datetime

import pickle

ROS_VERSION = os.environ.get("ROS_VERSION", "None")
USE_ROS = ROS_VERSION in ("1", "2")

print("Controller CARLA ROS VERSION:", ROS_VERSION)

try:
    import carla
except ModuleNotFoundError as ex:
    logger.error("CARLA is not supported")

from utils import metrics_carla
from utils.metrics_carla import build_json_summary, save_json_summary

# from std_srvs.srv import Empty
# from sensor_msgs.msg import Image as RosImage
# from cv_bridge import CvBridge
# from datetime import datetime
# from std_msgs.msg import String
#
# from utils.constants import CARLA_INFRACTION_PENALTIES
# try:
#     from carla_msgs.msg import CarlaLaneInvasionEvent
#     from carla_msgs.msg import CarlaCollisionEvent
# except ModuleNotFoundError as ex:
#     logger.error('CARLA is not supported')

from utils import metrics_carla

print(metrics_carla.__file__)
print(hasattr(metrics_carla, "get_metrics_python_api"))


# Stubs por defecto cuando no hay ROS
Node = object
RosImage = None
String = None
CvBridge = None
CarlaLaneInvasionEvent = None
CarlaCollisionEvent = None

if ROS_VERSION == "2":
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from sensor_msgs.msg import Image as RosImage

    try:
        from cv_bridge import CvBridge
    except Exception:
        CvBridge = None
    try:
        from carla_msgs.msg import CarlaLaneInvasionEvent, CarlaCollisionEvent
    except Exception:
        CarlaLaneInvasionEvent = None
        CarlaCollisionEvent = None

elif ROS_VERSION == "1":
    import rospy
    from std_msgs.msg import String
    from sensor_msgs.msg import Image as RosImage

    try:
        from cv_bridge import CvBridge
    except Exception:
        CvBridge = None
    try:
        from carla_msgs.msg import CarlaLaneInvasionEvent, CarlaCollisionEvent
    except Exception:
        CarlaLaneInvasionEvent = None
        CarlaCollisionEvent = None
else:
    CvBridge = None


from PIL import Image as PILImage


__author__ = "sergiopaniego"
__contributors__ = []
__license__ = "GPLv3"


# debug. function to convert numpy types to native python types
def convert_np_to_native(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_native(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_to_native(v) for v in obj)
    elif hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return obj.item()
    else:
        return obj


class ControllerCarla:
    """This class defines the controller of the architecture, responsible of the communication between the logic (model)
    and the user interface (view).

    Attributes:
        data {dict} -- Data to be sent to the view. The key is a frame_if of the view and the value is the data to be
        displayed. Depending on the type of data the frame handles (images, laser, etc)
        pose3D_data -- Pose data to be sent to the view
        recording {bool} -- Flag to determine if a rosbag is being recorded
    """

    def __init__(self, node):
        """Constructor of the class."""
        # pass
        self.node = node
        self.__data_loc = threading.Lock()
        self.__pose_loc = threading.Lock()
        self.data = {}
        self.pose3D_data = None
        self.recording = False
        self.cvbridge = CvBridge() if CvBridge else None

        self.rosbag_proc = None
        self.proc = None

        client = carla.Client("localhost", 2000)
        client.set_timeout(100.0)  # seconds
        
        self.csv_lock = threading.Lock()
        try:
            self.world = client.get_world()
        except RuntimeError as e:
            logger.warning("CARLA RuntimeError: {}".format(e))
        except carla.TimeoutException as e:
            logger.warning("CARLA timeout al cerrar {}".format(e))
            self.world = None

        time.sleep(10)  # takes a few second for the correct map to finish loading
        self.carla_map = self.world.get_map()

        while len(self.world.get_actors().filter("vehicle.*")) == 0:
            logger.info("Waiting for vehicles!")
            time.sleep(1)
        ego_vehicle_role_name = "ego_vehicle"
        self.ego_vehicle = None

        while self.ego_vehicle is None:
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name") == ego_vehicle_role_name:
                    self.ego_vehicle = vehicle
                    break
            if self.ego_vehicle is None:
                logger.info("Waiting for vehicle with role_name 'ego_vehicle'")
                time.sleep(1)  # sleep for 1 second before checking again
        # TODO: agregar solo waypoints de la ruta deseada
        self.map_waypoints = self.carla_map.generate_waypoints(0.5)
        self.weather = self.world.get_weather()

        # CSV Metrics from python API
        self.use_ros = ROS_VERSION in ("1", "2")

        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_headers = None
        self._tick_conn = None  # connection to tick event
        self._last_collision_impulse = 0.0
        self._last_collision_with = ""
        self._last_lane_invasion = ""

        # events sensors
        self._collision_actor = None
        self._lane_inavasion_actor = None
        try:
            bp_lib = self.world.get_blueprint_library()
            # collision sensor
            col_bp = bp_lib.find("sensor.other.collision")
            self._collision_actor = self.world.spawn_actor(
                col_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            weak_self = weakref.ref(self)
            self._collision_actor.listen(
                lambda e: ControllerCarla._on_collision(weak_self, e)
            )
            # lane invasion
            li_bp = bp_lib.find("sensor.other.lane_invasion")
            self._lane_invasion_actor = self.world.spawn_actor(
                li_bp, carla.Transform(), attach_to=self.ego_vehicle
            )
            self._lane_invasion_actor.listen(
                lambda e: ControllerCarla._on_lane_invasion(weak_self, e)
            )
        except Exception as e:
            logger.warning(f"Error setting up sensors: {e}")

    # GUI update
    def update_frame(self, frame_id, data):
        """Update the data to be retrieved by the view.

        This function is called by the logic to update the data obtained by the robot to a specific frame in GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame that will show the data
            data {dict} -- Data to be shown
        """
        try:
            with self.__data_loc:
                self.data[frame_id] = data
        except Exception as e:
            logger.info("Error updating frame {}: {}".format(frame_id, e))

    def get_data(self, frame_id):
        """Function to collect data retrieved by the robot for an specific frame of the GUI

        This function is called by the view to get the last updated data to be shown in the GUI.

        Arguments:
            frame_id {str} -- Identifier of the frame.

        Returns:
            data -- Depending on the caller frame could be image data, laser data, etc.
        """
        try:
            with self.__data_loc:
                data = self.data.get(frame_id, None)
        except Exception:
            pass

        return data

    def update_pose3d(self, data):
        """Update the pose3D data retrieved from the robot

        Arguments:
            data {pose3d} -- 3D position of the robot in the environment
        """
        try:
            with self.__pose_loc:
                self.pose3D_data = data
        except Exception:
            pass

    def get_pose3D(self):
        """Function to collect the pose3D data updated in `update_pose3d` function.

        This method is called from the view to collect the pose data and display it in GUI.

        Returns:
            pose3d -- 3D position of the robot in the environment
        """
        return self.pose3D_data

    # Simulation and dataset
    def reset_carla_simulation(self):
        logger.info("Restarting simulation")

    def pause_carla_simulation(self):
        logger.info("Pausing simulation")
        self.pilot.stop_event.set()

    def unpause_carla_simulation(self):
        logger.info("Resuming simulation")
        self.pilot.stop_event.clear()

    def record_rosbag(self, topics, dataset_name):
        """Start the recording process of the dataset using rosbags

        Arguments:
            topics {list} -- List of topics to be recorde
            dataset_name {str} -- Path of the resulting bag file
        """
        if self.recording:
            logger.info("Rosbag already recording")
            self.stop_record()
            return

        self.recording = True

        if ROS_VERSION == "2":
            command = (
                "ros2 bag record -o "
                + dataset_name
                + "/behav_bag"
                + " "
                + " ".join(topics)
            )
        else:
            command = (
                "rosbag record -O "
                + dataset_name
                + " "
                + " ".join(topics)
                + " __name:=behav_bag"
            )

        logger.info("Recording bag at: {}".format(dataset_name))
        cmd_split = shlex.split(command)
        with open("./logs/.roslaunch_stdout.log", "w") as out, open(
            "./logs/.roslaunch_stderr.log", "w"
        ) as err:
            self.rosbag_proc = subprocess.Popen(cmd_split, stdout=out, stderr=err)

    def stop_record(self):
        """Stop the rosbag recording process."""
        if not self.recording or not self.rosbag_proc:
            logger.info("No bag recording")
            return

        if ROS_VERSION == "2":
            self.rosbag_proc.terminate()
            self.rosbag_proc.wait()
            logger.info("Stopped bag recording")
        else:
            command = "rosnode kill /behav_bag"
            command_split = shlex.split(command)
            with open("./logs/.roslaunch_stdout.log", "w") as out, open(
                "./logs/.roslaunch_stderr.log", "w"
            ) as err:
                subprocess.Popen(command_split, stdout=out, stderr=err)
        self.recording = False
        self.rosbag_proc = None

    def reload_brain(self, brain, model=None):
        """Helper function to reload the current brain from the GUI.

        Arguments:
            brain {srt} -- Brain to be reloadaed.
        """
        logger.info("Reloading brain... {}".format(brain))

        self.pause_pilot()
        self.pilot.reload_brain(brain, model)

    # Helper functions (connection with logic)

    def set_pilot(self, pilot):
        self.pilot = pilot

    def stop_pilot(self):
        self.pilot.kill_event.set()

    def pause_pilot(self):
        self.pilot.stop_event.set()

    def resume_pilot(self):
        self.start_time = datetime.now()
        self.pilot.start_time = datetime.now()
        self.pilot.stop_event.clear()

    def initialize_robot(self):
        self.pause_pilot()
        self.pilot.initialize_robot()

    def record_metrics(self, metrics_record_dir_path, world_counter=None, brain_counter=None, repetition_counter=None):
        logger.info("Recording metrics: {}".format(metrics_record_dir_path))

        self.pilot.brain_iterations_real_time = []
        self.time_str = time.strftime("%Y%m%d-%H%M%S")

        if world_counter is not None:
            current_world_head, current_world_tail = os.path.split(
                self.pilot.configuration.current_world[world_counter]
            )
        else:
            current_world_head, current_world_tail = os.path.split(
                self.pilot.configuration.current_world
            )
        if brain_counter is not None:
            current_brain_head, current_brain_tail = os.path.split(
                self.pilot.configuration.brain_path[brain_counter]
            )
        else:
            current_brain_head, current_brain_tail = os.path.split(
                self.pilot.configuration.brain_path
            )

        self.experiment_metrics = {
            "timestamp": self.time_str,
            "experiment_configuration": self.pilot.configuration.__dict__,
            "world_launch_file": current_world_tail,
            "brain_file": current_brain_tail,
            "robot_type": self.pilot.configuration.robot_type,
            "carla_map": self.carla_map.name if self.carla_map else "",
            "ego_vehicle": (self.ego_vehicle.type_id if self.ego_vehicle else ""),
            "vehicles_number": (
                len(self.world.get_actors().filter("vehicle.*")) if self.world else 0
            ),
            "async_mode": self.pilot.configuration.async_mode,
            "weather": {
                "cloudiness": self.weather.cloudiness,
                "precipitation": self.weather.precipitation,
                "precipitation_deposits": self.weather.precipitation_deposits,
                "wind_intensity": self.weather.wind_intensity,
                "sun_azimuth_angle": self.weather.sun_azimuth_angle,
                "sun_altitude_angle": self.weather.sun_altitude_angle,
                "fog_density": self.weather.fog_density,
                "fog_distance": self.weather.fog_distance,
                "fog_falloff": self.weather.fog_falloff,
                "wetness": self.weather.wetness,
                "scattering_intensity": self.weather.scattering_intensity,
                "mie_scattering_scale": self.weather.mie_scattering_scale,
                "rayleigh_scattering_scale": self.weather.rayleigh_scattering_scale,
            },
        }

        if hasattr(self.pilot.configuration, "experiment_model"):
            self.experiment_metrics["experiment_model"] = (
                self.pilot.configuration.experiment_model[brain_counter]
                if brain_counter is not None
                else self.pilot.configuration.experiment_model
            )

        if hasattr(self.pilot.configuration, "experiment_name"):
            self.experiment_metrics["experiment_name"] = (
                self.pilot.configuration.experiment_name
            )
            self.experiment_metrics["experiment_description"] = getattr(
                self.pilot.configuration, "experiment_description", ""
            )
            if (
                hasattr(self.pilot.configuration, "experiment_timeouts")
                and world_counter is not None
            ):
                self.experiment_metrics["experiment_timeout"] = (
                    self.pilot.configuration.experiment_timeouts[world_counter]
                )
            self.experiment_metrics["experiment_repetition"] = repetition_counter

        self.metrics_record_dir_path = metrics_record_dir_path
        os.makedirs(
            os.path.join(self.metrics_record_dir_path, self.time_str), exist_ok=True
        )
        self.experiment_metrics_bag_filename = os.path.join(
            self.metrics_record_dir_path, self.time_str, self.time_str
        )

        # selection backend recording metrics
        if not hasattr(self, "use_ros"):
            ros_version_local = os.environ.get("ROS_VERSION", "2")
            self.use_ros = ros_version_local in ("1", "2")

        if self.use_ros:
            # backend ROS
            topics = [
                "/carla/npc_vehicle_1/odometry",
                "/carla/ego_vehicle/odometry",
                "/carla/ego_vehicle/collision",
                "/carla/ego_vehicle/lane_invasion",
                "/carla/ego_vehicle/speedometer",
                "/carla/ego_vehicle/vehicle_status",
                "/clock",
                "/carla/ego_vehicle/rgb_front/image",
            ]
            if os.environ.get("ROS_VERSION", "2") == "2":
                command = (
                    "ros2 bag record -o "
                    + self.experiment_metrics_bag_filename
                    + " "
                    + " ".join(topics)
                )
            else:
                command = (
                    "rosbag record -O "
                    + self.experiment_metrics_bag_filename
                    + " "
                    + " ".join(topics)
                    + " __name:=behav_metrics_bag"
                )

            cmd = shlex.split(command)
            with open("./logs/.roslaunch_stdout.log", "w") as out, open(
                "./logs/.roslaunch_stderr.log", "w"
            ) as err:
                logger.info(
                    f"Starting metrics bag recording with command: {' '.join(cmd)}"
                )
                self.proc = subprocess.Popen(cmd, stdout=out, stderr=err)
            logger.info("Started metrics bag recording")

        else:
            # backend Python API
            if not all(hasattr(self, name) for name in ("_csv_open", "_attach_tick")):
                raise RuntimeError(
                    "Faltan helpers CSV (_csv_open/_attach_tick). Add before use Python API metrics recording."
                )

            self._csv_open(self.metrics_record_dir_path)

            experiment_json_meta = os.path.join(
                self.metrics_record_dir_path,
                self.time_str,
                self.time_str + ".meta.json",
            )
            with open(experiment_json_meta, "w") as f:
                json.dump(convert_np_to_native(self.experiment_metrics), f)

            self._attach_tick()

            logger.info("Started CSV metrics recording (Python API)")

    def stop_recording_metrics(self, termination_code=None, route_length=None):
        logger.info("Stopping metrics recording")
        end_time = time.time()

        if not hasattr(self, "use_ros"):
            ros_version_local = os.environ.get("ROS_VERSION", "2")
            self.use_ros = ros_version_local in ("1", "2")

        if self.use_ros:
            # ros bag backend
            if os.environ.get("ROS_VERSION", "2") == "2":
                if self.proc:
                    self.proc.terminate()
                    self.proc.wait()
                    logger.info("Stopped bag recording")
            else:
                command = "rosnode kill /behav_metrics_bag"
                cmd = shlex.split(command)
                with open("./logs/.roslaunch_stdout.log", "w") as out, open(
                    "./logs/.roslaunch_stderr.log", "w"
                ) as err:
                    subprocess.Popen(cmd, stdout=out, stderr=err)

            timeout_counter = 20
            bag_active_file = self.experiment_metrics_bag_filename + ".active"
            while os.path.isfile(bag_active_file) and timeout_counter > 0:
                time.sleep(1)
                timeout_counter -= 1
            if timeout_counter <= 0:
                logger.warning(f"Timeout: {bag_active_file} not removed in time.")

            experiment_metrics_filename = os.path.join(
                self.metrics_record_dir_path, self.time_str, self.time_str
            )
            try:
                self.experiment_metrics = metrics_carla.get_metrics(
                    self.experiment_metrics,
                    self.experiment_metrics_bag_filename,
                    self.map_waypoints,
                    experiment_metrics_filename,
                    self.pilot.configuration,
                )
            except Exception as e:
                logger.error(f"Error while processing metrics (ROS): {e}")
                self.experiment_metrics = {}

        else:
            # python api backend
            if not all(hasattr(self, name) for name in ("_detach_tick", "_csv_close")):
                raise RuntimeError(
                    "Faltan helpers CSV (_detach_tick/_csv_close). Añádelos antes de usar el modo Python API."
                )

            try:
                self._detach_tick()
            finally:
                self._csv_close()

            experiment_metrics_filename = os.path.join(
                self.metrics_record_dir_path, self.time_str, self.time_str
            )
            try:
                self.experiment_metrics = metrics_carla.get_metrics_python_api(
                    self.experiment_metrics,
                    csv_path=self.csv_path,
                    map_waypoints=self.map_waypoints,
                    experiment_metrics_filename=experiment_metrics_filename,
                    config=self.pilot.configuration,
                )
            except Exception as e:
                logger.error(f"Error while processing metrics (CSV): {e}")
                self.experiment_metrics = {}
        try:
            self.experiment_metrics["experiment_total_real_time"] = (
                end_time - self.pilot.pilot_start_time
            )
        except Exception:
            self.experiment_metrics["experiment_total_real_time"] = (
                end_time - time.time()
            )

        experiment_json_path = os.path.join(
            self.metrics_record_dir_path, self.time_str, self.time_str + ".json"
        )
        os.makedirs(os.path.dirname(experiment_json_path), exist_ok=True)
        # with open(experiment_json_path, "w") as f:
            # try:
        self.experiment_metrics["experiment_total_real_time"] = (
            # end_time - self.pilot.pilot_start_time
            end_time - getattr(self.pilot, "pilot_start_time", end_time)
        )
        # except Exception:
        #     self.experiment_metrics["experiment_total_real_time"] = (
        #         end_time - time.time()
        #     )

        # Ruta base (igual que antes)
        experiment_json_path = os.path.join(
            self.metrics_record_dir_path, self.time_str, self.time_str + ".json"
        )
        os.makedirs(os.path.dirname(experiment_json_path), exist_ok=True)

        #
        summary = build_json_summary(
            experiment_metrics=self.experiment_metrics,
            extras={
                "timestamp": self.time_str,
                "world": getattr(self.pilot.configuration, "current_world", "N/A"),
                "brain": getattr(self.pilot.configuration, "brain_path", "N/A"),
                "experiment_name": getattr(
                    self.pilot.configuration, "experiment_name", "N/A"
                ),
                # "gpu_inference": getattr(self.pilot.brain, "gpu_inference", getattr(self, "use_gpu", "N/A")),
            },
        )

        # Guardar
        save_json_summary(experiment_json_path, summary)

        logger.info(f"Metrics stored in JSON file: {experiment_json_path}")
        logger.info("Stopped metrics recording")

        logger.info(f"Metrics stored in JSON file: {experiment_json_path}")
        logger.info("Stopped metrics recording")

    def save_metrics(self, first_images, last_images):
        with open(
            self.metrics_record_dir_path
            + self.time_str
            + "/"
            + self.time_str
            + ".json",
            "w",
        ) as f:
            json.dump(self.experiment_metrics, f)
        logger.info("Metrics stored in JSON file")

        for counter, image in enumerate(first_images):
            im = PILImage.fromarray(image)
            im.save(
                self.metrics_record_dir_path
                + self.time_str
                + "/"
                + self.time_str
                + "_first_image_"
                + str(counter)
                + ".jpeg"
            )

        for counter, image in enumerate(last_images):
            im = PILImage.fromarray(image)
            im.save(
                self.metrics_record_dir_path
                + self.time_str
                + "/"
                + self.time_str
                + "_last_image_"
                + str(counter)
                + ".jpeg"
            )

    # python API csv metrics helpers
    def _metrics_headers(self):
        return [
            "time_stamp",
            "frame",
            "x",
            "y",
            "z",
            "yaw",
            "pitch",
            "roll",
            "vx",
            "vy",
            "vz",
            "speed_mps",
            "speed_kmh",
            "throttle",
            "steer",
            "brake",
            "reverse",
            "gear",
            "collision_impulse",
            "collision_with",
            "lane_invasion",
            "weather_cloud",
            "weather_rain",
            "weather_wetness",
            "weather_fog",
        ]

    def _sample_row(self):
        # Time/Frame (if not synced, use world.get_snapshot())
        snap = self.world.get_snapshot()
        ts = snap.timestamp.elapsed_seconds if snap else time.time()
        frame = snap.frame if snap else -1

        t = self.ego_vehicle.get_transform()
        v = self.ego_vehicle.get_velocity()
        ctrl = self.ego_vehicle.get_control()

        speed_mps = math.sqrt(v.x**2 + v.y**2 + v.z**2)
        speed_kmh = 3.6 * speed_mps

        w = self.world.get_weather()

        return [
            ts,
            frame,
            t.location.x,
            t.location.y,
            t.location.z,
            t.rotation.yaw,
            t.rotation.pitch,
            t.rotation.roll,
            v.x,
            v.y,
            v.z,
            speed_mps,
            speed_kmh,
            getattr(ctrl, "throttle", 0.0),
            getattr(ctrl, "steer", 0.0),
            getattr(ctrl, "brake", 0.0),
            bool(getattr(ctrl, "reverse", False)),
            getattr(ctrl, "gear", 0),
            self._last_collision_impulse,
            self._last_collision_with,
            self._last_lane_invasion,
            w.cloudiness,
            w.precipitation,
            w.wetness,
            w.fog_density,
        ]


    def _csv_open(self, out_dir_base):
        os.makedirs(os.path.join(out_dir_base, self.time_str), exist_ok=True)
        self.csv_path = os.path.join(out_dir_base, self.time_str, f"{self.time_str}.csv")
        self.csv_headers = self._metrics_headers()
        with self.csv_lock:
            self.csv_file = open(self.csv_path, "w", buffering=1)  # line buffered
            self.csv_file.write(",".join(self.csv_headers) + "\n")
        logger.info(f"CSV metrics at: {self.csv_path}")


    def _csv_append(self, row_vals):
        line = ",".join(str(v) for v in convert_np_to_native(row_vals)) + "\n"
        with self.csv_lock:
            f = self.csv_file
            if not f:
                return
            try:
                f.write(line)
            except Exception as e:
                logger.warning(f"CSV append failed: {e}")


    def _csv_close(self):
        with self.csv_lock:
            f = self.csv_file
            self.csv_file = None  # invalida el handle para otros hilos antes de cerrar
        if f:
            try:
                f.flush()
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass


    def _on_tick_cb(self, snapshot):
        # Evita trabajo si el CSV ya no existe
        with self.csv_lock:
            if self.csv_file is None:
                return
        try:
            row = self._sample_row()
            self._csv_append(row)  # _csv_append vuelve a verificar con candado
        except Exception as e:
            logger.warning(f"Failed to sample CSV: {e}")

    def _attach_tick(self):
        if self._tick_conn is None:
            self._tick_conn = self.world.on_tick(self._on_tick_cb)

    def _detach_tick(self):
        if self._tick_conn is not None:
            # on_tick devuelve un objeto "event" que actúa como callable removible (en 0.9.15 es deregistrar con "None")
            try:
                self.world.remove_on_tick(self._tick_conn)  # si está disponible
            except Exception:
                # Fallback (0.9.1x no siempre expone remove_on_tick)
                self._tick_conn = None
            self._tick_conn = None

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        imp = event.normal_impulse
        self._last_collision_impulse = math.sqrt(imp.x**2 + imp.y**2 + imp.z**2)
        self._last_collision_with = event.other_actor.type_id

    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        marks = [str(x.type).split(".")[-1] for x in event.crossed_lane_markings]
        self._last_lane_invasion = "|".join(marks)
