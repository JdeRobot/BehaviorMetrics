#!/usr/bin/env python

"""This module contains the metrics manager.
This module is in charge of generating metrics for a brain execution.
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import time
import os

# import rosbag
import re

# from bagpy import bagreader
from utils.logger import logger

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math, json


def circuit_distance_completed(checkpoints, lap_point):
    previous_point = []
    diameter = 0
    for i, point in enumerate(checkpoints):
        current_point = np.array(
            [point["pose.pose.position.x"], point["pose.pose.position.y"]]
        )
        if i != 0:
            dist = (previous_point - current_point) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            diameter += dist
        if point is lap_point:
            break
        previous_point = current_point
    return diameter


def get_metrics(
    experiment_metrics,
    experiment_metrics_bag_filename,
    map_waypoints,
    experiment_metrics_filename,
    config,
):

    print("Experiment metrics bag filename:", experiment_metrics_bag_filename)

    time_counter = 5
    while not os.path.exists(experiment_metrics_bag_filename):
        time.sleep(1)
        time_counter -= 1
        if time_counter <= 0:
            ValueError(f"{experiment_metrics_bag_filename} isn't a file!")
            return {}

    ros_version = os.environ.get("ROS_VERSION", "2")

    bag_reader = None

    if ros_version == "2":
        try:
            import rosbag2_py
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message

            rosbag_dir = experiment_metrics_bag_filename

            storage_options = rosbag2_py.StorageOptions(
                uri=rosbag_dir, storage_id="sqlite3"
            )
            converter_options = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
            bag_reader = reader

            topic_types = {
                t.name: t.type for t in bag_reader.get_all_topics_and_types()
            }
            # data_by_topic = {topic: [] for topic in topic_types.keys()}
            data_by_topic = {topic: [] for topic in topic_types}

            while bag_reader.has_next():
                topic, data, t = bag_reader.read_next()
                msg_type_str = topic_types[topic]
                msg_type = get_message(msg_type_str)
                msg = deserialize_message(data, msg_type)
                data_by_topic[topic].append(msg)

            print("Topics in bag:", list(data_by_topic.keys()))
            csv_files = data_by_topic

        except Exception as e:
            print("Error procesando mensajes del bag en ROS2:", e)
            return {}
    else:
        try:
            import rosbag

            bag_reader = rosbag.Bag(experiment_metrics_bag_filename)
        except rosbag.ROSBagException as e:
            print("Error reading bag file in ROS 1:", e)
            bag_reader = None

    required_topics = [
        "/carla/ego_vehicle/odometry",
        "/carla/ego_vehicle/speedometer",
        "/carla/ego_vehicle/vehicle_status",
        "/clock",
    ]
    if not all(topic in data_by_topic for topic in required_topics):
        print("Faltan tópicos necesarios en el bag.")
        return {}

    odometry_msgs = data_by_topic["/carla/ego_vehicle/odometry"]
    checkpoints = []
    for msg in odometry_msgs:
        checkpoints.append(
            {
                "pose.pose.position.x": msg.pose.pose.position.x,
                "pose.pose.position.y": msg.pose.pose.position.y,
                "Time": 0,
            }
        )
    dataframe_pose = pd.DataFrame(checkpoints)

    if config.task == "follow_lane_traffic":
        data_file = (
            experiment_metrics_bag_filename.split(".bag")[0]
            + "/carla-npc_vehicle_1-odometry.csv"
        )
        dataframe_pose = pd.read_csv(data_file)
        checkpoints_2 = []
        for index, row in dataframe_pose.iterrows():
            checkpoints_2.append(row)

    clock_points = []

    dataframe_clock = pd.DataFrame(
        [
            {"clock.secs": msg.clock.sec, "Time": i}
            for i, msg in enumerate(data_by_topic["/clock"])
        ]
    )
    clock_points = dataframe_clock.to_dict("records")
    start_clock = clock_points[0]
    seconds_start = start_clock["clock.secs"]
    seconds_end = clock_points[len(clock_points) - 1]["clock.secs"]

    collision_points = []
    if "carla/ego_vehicle/collision" in data_by_topic:
        dataframe_collision = pd.DataFrame(
            [
                {"Time": i}
                for i, msg in enumerate(data_by_topic["carla/ego_vehicle/collision"])
            ]
        )
        collision_points = dataframe_collision.to_dict("records")

    if "/carla/ego_vehicle/lane_invasion" in data_by_topic:
        dataframe_lane_invasion = pd.DataFrame(
            [
                {"Time": i}
                for i, msg in enumerate(
                    data_by_topic["/carla/ego_vehicle/lane_invasion"]
                )
            ]
        )
        lane_invasion_points = dataframe_lane_invasion.to_dict("records")

    speedometer_points = []
    dataframe_speedometer = pd.DataFrame(
        [
            {"data": msg.data, "Time": i}
            for i, msg in enumerate(data_by_topic["/carla/ego_vehicle/speedometer"])
        ]
    )
    speedometer_points = dataframe_speedometer.to_dict("records")

    vehicle_status_points = []
    dataframe_vehicle_status = pd.DataFrame(
        [
            {
                "control.throttle": msg.control.throttle,
                "control.steer": msg.control.steer,
                "control.brake": msg.control.brake,
            }
            for i, msg in enumerate(data_by_topic["/carla/ego_vehicle/vehicle_status"])
        ]
    )
    vehicle_status_points = dataframe_vehicle_status.to_dict("records")

    if map_waypoints:
        experiment_metrics = get_percentage_completed(
            experiment_metrics, checkpoints, map_waypoints
        )

    if len(checkpoints) > 1:
        starting_point = checkpoints[0]
        starting_point = (
            starting_point["pose.pose.position.x"],
            starting_point["pose.pose.position.y"],
        )
        experiment_metrics["starting_point"] = starting_point
        experiment_metrics = get_distance_completed(experiment_metrics, checkpoints)
        experiment_metrics = get_average_speed(experiment_metrics, speedometer_points)
        experiment_metrics = get_suddenness_control_commands(
            experiment_metrics, vehicle_status_points
        )
        experiment_metrics, collisions_checkpoints = get_collisions(
            experiment_metrics, collision_points, dataframe_pose
        )
        experiment_metrics, lane_invasion_checkpoints = get_lane_invasions(
            experiment_metrics, lane_invasion_points, dataframe_pose
        )
        experiment_metrics["experiment_total_simulated_time"] = (
            seconds_end - seconds_start
        )
        if config.task == "follow_lane_traffic":
            experiment_metrics = get_distance_other_vehicle(
                experiment_metrics, checkpoints, checkpoints_2
            )

        if "bird_eye_view_images" in experiment_metrics:
            experiment_metrics["bird_eye_view_images_per_second"] = (
                experiment_metrics["bird_eye_view_images"]
                / experiment_metrics["experiment_total_simulated_time"]
            )
            experiment_metrics["bird_eye_view_unique_images_per_second"] = (
                experiment_metrics["bird_eye_view_unique_images"]
                / experiment_metrics["experiment_total_simulated_time"]
            )

        experiment_metrics = get_position_deviation_and_effective_completed_distance(
            experiment_metrics,
            checkpoints,
            map_waypoints,
            experiment_metrics_filename,
            speedometer_points,
            collisions_checkpoints,
            lane_invasion_checkpoints,
        )
        experiment_metrics["completed_laps"] = get_completed_laps(
            checkpoints, starting_point
        )
        # shutil.rmtree(experiment_metrics_bag_filename.split('.bag')[0]) # DEBUG
        return experiment_metrics
    else:
        return {}


def get_completed_laps(checkpoints, starting_point):
    points_to_start_count = 50
    completed_laps = 0
    for x, checkpoint in enumerate(checkpoints):
        if points_to_start_count > 0:
            points_to_start_count -= 1
        else:
            point_1 = np.array(
                [checkpoint["pose.pose.position.x"], checkpoint["pose.pose.position.y"]]
            )
            point_2 = np.array([starting_point[0], starting_point[1]])
            dist = (point_2 - point_1) ** 2
            dist = np.sum(dist, axis=0)
            dist = np.sqrt(dist)
            if dist < 0.5:
                completed_laps += 1
                points_to_start_count = 50

    return completed_laps


def get_distance_completed(experiment_metrics, checkpoints):
    end_point = checkpoints[len(checkpoints) - 1]
    experiment_metrics["completed_distance"] = circuit_distance_completed(
        checkpoints, end_point
    )
    return experiment_metrics


def get_percentage_completed(experiment_metrics, checkpoints, map_waypoints):
    # Covert waypoints (CARLA waypoitnts) into 'perfect_lap_checkpoints' structure
    perfect_lap_checkpoints = [
        {
            "pose.pose.position.x": wp.transform.location.x,
            "pose.pose.position.y": wp.transform.location.y,
        }
        for wp in map_waypoints
    ]

    # starting_point = checkpoints[0]
    first_checkpoint = np.array(
        [checkpoints[0]["pose.pose.position.x"], checkpoints[0]["pose.pose.position.y"]]
    )

    # found nearest point to the first checkpoint
    perfect_point_iterator = 0
    min_dist = float("inf")
    for position, perfect_checkpoint in enumerate(perfect_lap_checkpoints):
        perfect_pos = np.array(
            [
                perfect_checkpoint["pose.pose.position.x"],
                perfect_checkpoint["pose.pose.position.y"],
            ]
        )
        dist = np.linalg.norm(perfect_pos - first_checkpoint)
        if dist < min_dist:
            min_dist = dist
            perfect_point_iterator = position

    lap_checkpoint = 0  # for round completed register

    # Direction 1
    checkpoints_reached_dir_1 = 1
    checkpoint_iterator = 1
    perfect_point_iterator_dir_1 = perfect_point_iterator + 1

    while checkpoint_iterator < len(checkpoints):
        if perfect_point_iterator_dir_1 >= len(perfect_lap_checkpoints):
            perfect_point_iterator_dir_1 = 0
        current_checkpoint = np.array(
            [
                checkpoints[checkpoint_iterator]["pose.pose.position.x"],
                checkpoints[checkpoint_iterator]["pose.pose.position.y"],
            ]
        )
        perfect_checkpoint = np.array(
            [
                perfect_lap_checkpoints[perfect_point_iterator_dir_1][
                    "pose.pose.position.x"
                ],
                perfect_lap_checkpoints[perfect_point_iterator_dir_1][
                    "pose.pose.position.y"
                ],
            ]
        )
        dist = np.linalg.norm(perfect_checkpoint - current_checkpoint)
        if dist < 5:
            checkpoints_reached_dir_1 += 1
            perfect_point_iterator_dir_1 += 1
            if checkpoints_reached_dir_1 == len(perfect_lap_checkpoints):
                lap_checkpoint = checkpoint_iterator
        else:
            checkpoint_iterator += 1

    percentage_completed_dir_1 = (
        checkpoints_reached_dir_1 / len(perfect_lap_checkpoints)
    ) * 100

    # Direction 2
    checkpoints_reached_dir_2 = 1
    checkpoint_iterator = 1
    perfect_point_iterator_dir_2 = perfect_point_iterator - 1

    while checkpoint_iterator < len(checkpoints):
        if perfect_point_iterator_dir_2 < 0:
            perfect_point_iterator_dir_2 = len(perfect_lap_checkpoints) - 1
        current_checkpoint = np.array(
            [
                checkpoints[checkpoint_iterator]["pose.pose.position.x"],
                checkpoints[checkpoint_iterator]["pose.pose.position.y"],
            ]
        )
        perfect_checkpoint = np.array(
            [
                perfect_lap_checkpoints[perfect_point_iterator_dir_2][
                    "pose.pose.position.x"
                ],
                perfect_lap_checkpoints[perfect_point_iterator_dir_2][
                    "pose.pose.position.y"
                ],
            ]
        )
        dist = np.linalg.norm(perfect_checkpoint - current_checkpoint)
        if dist < 5:
            checkpoints_reached_dir_2 += 1
            perfect_point_iterator_dir_2 -= 1
            if checkpoints_reached_dir_2 == len(perfect_lap_checkpoints):
                lap_checkpoint = checkpoint_iterator
        else:
            checkpoint_iterator += 1

    percentage_completed_dir_2 = (
        checkpoints_reached_dir_2 / len(perfect_lap_checkpoints)
    ) * 100

    # Select the maximum percentage completed
    experiment_metrics["percentage_completed"] = max(
        percentage_completed_dir_1, percentage_completed_dir_2
    )
    experiment_metrics["lap_checkpoint"] = lap_checkpoint
    return experiment_metrics


def get_average_speed(experiment_metrics, speedometer_points):
    previous_speed = 0
    speedometer_points_sum = 0
    suddenness_distance_speeds = []
    speed_points = []
    for point in speedometer_points:
        # speed_point = point.data*3.6
        speed_point = point["data"] * 3.6
        speedometer_points_sum += speed_point
        a = np.array(speed_point)
        b = np.array(previous_speed)
        suddenness_distance_speed = np.linalg.norm(a - b)
        suddenness_distance_speeds.append(suddenness_distance_speed)
        previous_speed = speed_point
        speed_points.append(speed_point)

    experiment_metrics["average_speed"] = speedometer_points_sum / len(
        speedometer_points
    )
    suddenness_distance_speed = sum(suddenness_distance_speeds) / len(
        suddenness_distance_speeds
    )
    experiment_metrics["suddenness_distance_speed"] = suddenness_distance_speed
    experiment_metrics["max_speed"] = max(speed_points)
    experiment_metrics["min_speed"] = min(speed_points)
    return experiment_metrics


def get_suddenness_control_commands(experiment_metrics, vehicle_status_points):
    previous_commanded_throttle = 0
    previous_commanded_steer = 0
    previous_commanded_brake = 0
    suddenness_distance_control_commands = []
    suddenness_distance_throttle = []
    suddenness_distance_steer = []
    suddenness_distance_brake_command = []

    for point in vehicle_status_points:
        throttle = point["control.throttle"]
        steer = point["control.steer"]
        brake_command = point["control.brake"]

        a = np.array((throttle, steer, brake_command))
        b = np.array(
            (
                previous_commanded_throttle,
                previous_commanded_steer,
                previous_commanded_brake,
            )
        )
        distance = np.linalg.norm(a - b)
        suddenness_distance_control_commands.append(distance)

        a = np.array((throttle))
        b = np.array((previous_commanded_throttle))
        distance_throttle = np.linalg.norm(a - b)
        suddenness_distance_throttle.append(distance_throttle)

        a = np.array((steer))
        b = np.array((previous_commanded_steer))
        distance_steer = np.linalg.norm(a - b)
        suddenness_distance_steer.append(distance_steer)

        a = np.array((brake_command))
        b = np.array((previous_commanded_brake))
        distance_brake_command = np.linalg.norm(a - b)
        suddenness_distance_brake_command.append(distance_brake_command)

        previous_commanded_throttle = throttle
        previous_commanded_steer = steer
        previous_commanded_brake = brake_command

    experiment_metrics["suddenness_distance_control_commands"] = sum(
        suddenness_distance_control_commands
    ) / len(suddenness_distance_control_commands)
    experiment_metrics["suddenness_distance_throttle"] = sum(
        suddenness_distance_throttle
    ) / len(suddenness_distance_throttle)
    experiment_metrics["suddenness_distance_steer"] = sum(
        suddenness_distance_steer
    ) / len(suddenness_distance_steer)
    experiment_metrics["suddenness_distance_brake_command"] = sum(
        suddenness_distance_brake_command
    ) / len(suddenness_distance_brake_command)
    return experiment_metrics


def get_collisions(experiment_metrics, collision_points, df_checkpoints):
    collisions_checkpoints = []
    collisions_checkpoints_different = []
    collisions_actors_different = []
    previous_collisions_checkpoints_x, previous_collisions_checkpoints_y = 0, 0
    for point in collision_points:
        idx = (df_checkpoints["Time"] - point["Time"]).abs().idxmin()
        collision_point = df_checkpoints.iloc[idx]
        collisions_checkpoints.append(collision_point)
        point_1 = np.array(
            [
                collision_point["pose.pose.position.x"],
                collision_point["pose.pose.position.y"],
            ]
        )
        point_2 = np.array(
            [previous_collisions_checkpoints_x, previous_collisions_checkpoints_y]
        )
        dist = (point_2 - point_1) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist > 1:
            collisions_checkpoints_different.append(collision_point)
            collisions_actors_different.append(point["other_actor_id"])
        previous_collisions_checkpoints_x, previous_collisions_checkpoints_y = (
            collision_point["pose.pose.position.x"],
            collision_point["pose.pose.position.y"],
        )

    experiment_metrics["collisions"] = len(collisions_checkpoints_different)
    experiment_metrics["collision_actor_ids"] = collisions_actors_different
    return experiment_metrics, collisions_checkpoints


def get_lane_invasions(experiment_metrics, lane_invasion_points, df_checkpoints):
    lane_invasion_checkpoints = []
    lane_invasion_checkpoints_different = []
    previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y = 0, 0
    previous_time = 0
    for point in lane_invasion_points:
        idx = (df_checkpoints["Time"] - point["Time"]).abs().idxmin()
        lane_invasion_point = df_checkpoints.iloc[idx]

        lane_invasion_checkpoints.append(lane_invasion_point)
        point_1 = np.array(
            [
                lane_invasion_point["pose.pose.position.x"],
                lane_invasion_point["pose.pose.position.y"],
            ]
        )
        point_2 = np.array(
            [previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y]
        )
        dist = (point_2 - point_1) ** 2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        if dist > 1 and point["Time"] - previous_time > 0.5:
            lane_invasion_checkpoints_different.append(lane_invasion_point)
        previous_time = point["Time"]
        previous_lane_invasion_checkpoints_x, previous_lane_invasion_checkpoints_y = (
            lane_invasion_point["pose.pose.position.x"],
            lane_invasion_point["pose.pose.position.y"],
        )

    experiment_metrics["lane_invasions"] = len(lane_invasion_checkpoints_different)
    return experiment_metrics, lane_invasion_checkpoints


def get_position_deviation_and_effective_completed_distance(
    experiment_metrics,
    checkpoints,
    map_waypoints,
    experiment_metrics_filename,
    speedometer,
    collision_points,
    lane_invasion_checkpoints,
):
    """
    Alinea la trayectoria con el mapa usando SOLO TRASLACIÓN (sin rotación ni reflexión).
    Calcula métricas espaciales y genera PNGs (general + específicos).
    Requiere:
      - create_experiment_maps(...)
      - create_collisions_map(...)
      - create_lane_invasions_map(...)
    """

    # --------- Helpers (traslación únicamente) ----------
    def _best_translation(checkpoints_xy, map_xy, method="centroid"):
        """
        Devuelve vector de traslación t=(dx,dy) para alinear checkpoints -> mapa.
        - method="centroid": t = mean(map) - mean(checkpoints)  (robusto y simple)
        - method="first_to_nearest": alinea el primer punto al waypoint más cercano
        """
        import numpy as np

        if not checkpoints_xy or not map_xy:
            return (0.0, 0.0)

        A = np.asarray(checkpoints_xy, float)
        B = np.asarray(map_xy, float)

        if method == "first_to_nearest":
            a0 = A[0]
            d2 = ((B - a0) ** 2).sum(axis=1)
            b_near = B[np.argmin(d2)]
            t = (float(b_near[0] - a0[0]), float(b_near[1] - a0[1]))
            return t
        else:
            ca = A.mean(axis=0)
            cb = B.mean(axis=0)
            t = (float(cb[0] - ca[0]), float(cb[1] - ca[1]))
            return t

    def _apply_translation(points, t):
        dx, dy = t
        return [(float(x) + dx, float(y) + dy) for x, y in points]


    # --------------- SIN MAPA: modo ligero ----------------
    if not map_waypoints:
        xs = (
            [float(p["pose.pose.position.x"]) for p in checkpoints]
            if checkpoints
            else []
        )
        ys = (
            [float(p["pose.pose.position.y"]) for p in checkpoints]
            if checkpoints
            else []
        )

        experiment_metrics.setdefault(
            "effective_completed_distance",
            float(experiment_metrics.get("completed_distance", 0.0)),
        )
        experiment_metrics.setdefault("position_deviation_mean", 0.0)
        experiment_metrics.setdefault("position_deviation_total_err", 0.0)
        km = max(experiment_metrics["effective_completed_distance"] / 1000.0, 1e-9)
        experiment_metrics.setdefault(
            "position_deviation_mean_per_km",
            experiment_metrics["position_deviation_mean"] / km,
        )
        experiment_metrics["starting_point_map"] = (xs[0], ys[0]) if xs else (0.0, 0.0)

        for k in [
            "collisions",
            "lane_invasions",
            "suddenness_distance_control_commands",
            "suddenness_distance_throttle",
            "suddenness_distance_steer",
            "suddenness_distance_brake_command",
            "suddenness_distance_speed",
        ]:
            if k in experiment_metrics:
                experiment_metrics[f"{k}_per_km"] = experiment_metrics[k] / km

        return experiment_metrics

    # --------------- MAPA: recopilar waypoints ----------------
    map_waypoints_tuples_x, map_waypoints_tuples_y, map_waypoints_tuples = [], [], []
    for wp in map_waypoints:
        wx = float(wp.transform.location.x)
        wy = float(wp.transform.location.y)
        map_waypoints_tuples_x.append(wx)
        map_waypoints_tuples_y.append(wy)
        map_waypoints_tuples.append((wx, wy))
    map_xy = list(zip(map_waypoints_tuples_x, map_waypoints_tuples_y))

    # --------------- Trayectoria + velocidades ----------------
    N = min(len(checkpoints), len(speedometer)) if speedometer else len(checkpoints)
    raw_xy, speeds_kmh = [], []
    for i in range(N):
        px = float(checkpoints[i]["pose.pose.position.x"])
        py = float(checkpoints[i]["pose.pose.position.y"])
        v_kmh = (
            float(speedometer[i]["data"]) * 3.6
            if speedometer and "data" in speedometer[i]
            else 0.0
        )
        raw_xy.append((px, py))
        speeds_kmh.append(v_kmh)

    # --------------- Alineación por TRASLACIÓN (sin rotación) ----------------
    #forzar inicio: method="first_to_nearest"
    t = _best_translation(raw_xy, map_xy, method="first_to_nearest")
    aligned_xy = _apply_translation(raw_xy, t)

    checkpoints_tuples_x = [x for x, _ in aligned_xy]
    checkpoints_tuples_y = [y for _, y in aligned_xy]
    checkpoints_tuples = [(x, y, v) for (x, y), v in zip(aligned_xy, speeds_kmh)]

    # Alinea también colisiones e invasiones
    aligned_collisions = _apply_translation(
        [
            (float(p["pose.pose.position.x"]), float(p["pose.pose.position.y"]))
            for p in (collision_points or [])
        ],
        t,
    )
    aligned_lane_inv = _apply_translation(
        [
            (float(p["pose.pose.position.x"]), float(p["pose.pose.position.y"]))
            for p in (lane_invasion_checkpoints or [])
        ],
        t,
    )

    # --------------- Distancia a eje de vía (NN sobre mapa) ----------------
    min_dists, best_checkpoint_points_x, best_checkpoint_points_y = [], [], []
    covered_checkpoints = []
    for cx, cy, _ in checkpoints_tuples:
        best_d = float("inf")
        bx, by = cx, cy
        for mx, my in map_waypoints_tuples:
            d = math.hypot(mx - cx, my - cy)
            if d < best_d:
                best_d = d
                bx, by = mx, my
        best_checkpoint_points_x.append(bx)
        best_checkpoint_points_y.append(by)
        if best_d < 100.0:
            min_dists.append(best_d)
            if (not covered_checkpoints or covered_checkpoints[-1] != (bx, by)) and (
                best_d < 1.0
            ):
                covered_checkpoints.append((bx, by))

    # --------------- Métricas espaciales ----------------
    eff_dist = float(len(covered_checkpoints) * 0.5)  # ~0.5 m por waypoint cubierto
    experiment_metrics["effective_completed_distance"] = eff_dist
    if min_dists:
        pdm = float(sum(min_dists) / len(min_dists))
        pdt = float(sum(min_dists))
    else:
        pdm, pdt = 0.0, 0.0
    experiment_metrics["position_deviation_mean"] = pdm
    experiment_metrics["position_deviation_total_err"] = pdt
    km = max(eff_dist / 1000.0, 1e-9)
    experiment_metrics["position_deviation_mean_per_km"] = pdm / km
    experiment_metrics["starting_point_map"] = (
        (checkpoints_tuples_x[0], checkpoints_tuples_y[0])
        if checkpoints_tuples_x
        else (0.0, 0.0)
    )

    for k in [
        "collisions",
        "lane_invasions",
        "suddenness_distance_control_commands",
        "suddenness_distance_throttle",
        "suddenness_distance_steer",
        "suddenness_distance_brake_command",
        "suddenness_distance_speed",
    ]:
        if k in experiment_metrics:
            experiment_metrics[f"{k}_per_km"] = experiment_metrics[k] / km

    # --------------- Landmarks inicio/fin para marcar en el plot ---------------
    starting_point_landmark = 0
    if len(checkpoints_tuples_x) > 1:
        dx = dy = 0.0
        while (
            dx < 1
            and dy < 1
            and starting_point_landmark < len(checkpoints_tuples_x) - 1
        ):
            dx = abs(
                checkpoints_tuples_x[starting_point_landmark] - checkpoints_tuples_x[0]
            )
            dy = abs(
                checkpoints_tuples_y[starting_point_landmark] - checkpoints_tuples_y[0]
            )
            if dx < 1 and dy < 1:
                starting_point_landmark += 1

    finish_point_landmark = max(len(checkpoints_tuples_x) - 1, 0)
    if len(checkpoints_tuples_x) > 1:
        dx = dy = 0.0
        while dx < 1 and dy < 1 and finish_point_landmark > 0:
            dx = abs(
                checkpoints_tuples_x[finish_point_landmark] - checkpoints_tuples_x[-1]
            )
            dy = abs(
                checkpoints_tuples_y[finish_point_landmark] - checkpoints_tuples_y[-1]
            )
            if dx < 1 and dy < 1:
                finish_point_landmark -= 1

    # Dibujos, dos imágenes
    create_experiment_maps(
        experiment_metrics,
        experiment_metrics_filename,
        map_waypoints_tuples_x,
        map_waypoints_tuples_y,
        best_checkpoint_points_x,
        best_checkpoint_points_y,
        checkpoints_tuples_x,
        checkpoints_tuples_y,
        speeds_kmh,
        [
            {"pose.pose.position.x": x, "pose.pose.position.y": y}
            for x, y in aligned_collisions
        ],
        [
            {"pose.pose.position.x": x, "pose.pose.position.y": y}
            for x, y in aligned_lane_inv
        ],
    )

    if experiment_metrics.get("collisions", 0) > 0 and aligned_collisions:
        create_collisions_map(
            experiment_metrics,
            experiment_metrics_filename,
            map_waypoints_tuples_x,
            map_waypoints_tuples_y,
            checkpoints_tuples_x,
            checkpoints_tuples_y,
            speeds_kmh,
            starting_point_landmark,
            finish_point_landmark,
            [
                {"pose.pose.position.x": x, "pose.pose.position.y": y}
                for x, y in aligned_collisions
            ],
        )

    if experiment_metrics.get("lane_invasions", 0) > 0 and aligned_lane_inv:
        create_lane_invasions_map(
            experiment_metrics,
            experiment_metrics_filename,
            map_waypoints_tuples_x,
            map_waypoints_tuples_y,
            checkpoints_tuples_x,
            checkpoints_tuples_y,
            speeds_kmh,
            starting_point_landmark,
            finish_point_landmark,
            [
                {"pose.pose.position.x": x, "pose.pose.position.y": y}
                for x, y in aligned_lane_inv
            ],
        )

    return experiment_metrics


def create_experiment_maps(
    experiment_metrics,
    experiment_metrics_filename,
    map_waypoints_tuples_x,
    map_waypoints_tuples_y,
    best_checkpoint_points_x,
    best_checkpoint_points_y,
    checkpoints_tuples_x,
    checkpoints_tuples_y,
    checkpoints_speeds,
    collision_points,
    lane_invasion_checkpoints,
):
    difference_x = 0
    difference_y = 0
    starting_point_landmark = 0
    while (
        difference_x < 1
        and difference_y < 1
        and starting_point_landmark < len(checkpoints_tuples_x) - 1
    ):
        difference_x = abs(
            checkpoints_tuples_x[starting_point_landmark] - checkpoints_tuples_x[0]
        )
        difference_y = abs(
            checkpoints_tuples_y[starting_point_landmark] - checkpoints_tuples_y[0]
        )
        if difference_x < 1 and difference_y < 1:
            starting_point_landmark += 1

    difference_x = 0
    difference_y = 0
    finish_point_landmark = len(checkpoints_tuples_x) - 1
    while difference_x < 1 and difference_y < 1 and finish_point_landmark > 0:
        difference_x = abs(
            checkpoints_tuples_x[finish_point_landmark]
            - checkpoints_tuples_x[len(checkpoints_tuples_x) - 1]
        )
        difference_y = abs(
            checkpoints_tuples_y[finish_point_landmark]
            - checkpoints_tuples_y[len(checkpoints_tuples_x) - 1]
        )
        if difference_x < 1 and difference_y < 1:
            finish_point_landmark -= 1

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    colors = ["#00FF00", "#FF0000", "#000000"]
    ax.scatter(
        map_waypoints_tuples_x,
        map_waypoints_tuples_y,
        s=10,
        c="b",
        marker="s",
        label="Map waypoints",
    )
    ax.scatter(
        best_checkpoint_points_x,
        best_checkpoint_points_y,
        s=10,
        c="g",
        marker="o",
        label="Map waypoints for position deviation",
    )
    plot = ax.scatter(
        checkpoints_tuples_x,
        checkpoints_tuples_y,
        s=10,
        c=checkpoints_speeds,
        cmap="hot_r",
        marker="o",
        label="Experiment waypoints",
        vmin=0,
        vmax=(
            experiment_metrics["max_speed"]
            if experiment_metrics["max_speed"] > 30
            else 30
        ),
    )
    ax.scatter(
        checkpoints_tuples_x[0],
        checkpoints_tuples_y[0],
        s=200,
        marker="o",
        color=colors[0],
        label="Experiment starting point",
    )
    ax.scatter(
        checkpoints_tuples_x[starting_point_landmark],
        checkpoints_tuples_y[starting_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    ax.scatter(
        checkpoints_tuples_x[len(checkpoints_tuples_x) - 1],
        checkpoints_tuples_y[len(checkpoints_tuples_x) - 1],
        s=200,
        marker="o",
        color=colors[1],
        label="Experiment finish point",
    )
    ax.scatter(
        checkpoints_tuples_x[finish_point_landmark],
        checkpoints_tuples_y[finish_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={"size": 20})

    full_text = ""
    for key, value in experiment_metrics.items():
        print(key, value)
        full_text += " * " + str(key) + " : " + str(value) + "\n"
    plt.figtext(
        0.1, 0.01, full_text, wrap=True, horizontalalignment="left", fontsize=10
    )

    plt.grid(True)
    plt.subplots_adjust(bottom=0.4)
    plt.title(experiment_metrics["experiment_model"], fontsize=16)
    fig.savefig(experiment_metrics_filename + ".png", dpi=fig.dpi)

    if experiment_metrics["collisions"] > 0:
        create_collisions_map(
            experiment_metrics,
            experiment_metrics_filename,
            map_waypoints_tuples_x,
            map_waypoints_tuples_y,
            checkpoints_tuples_x,
            checkpoints_tuples_y,
            checkpoints_speeds,
            starting_point_landmark,
            finish_point_landmark,
            collision_points,
        )
    if experiment_metrics["lane_invasions"] > 0:
        create_lane_invasions_map(
            experiment_metrics,
            experiment_metrics_filename,
            map_waypoints_tuples_x,
            map_waypoints_tuples_y,
            checkpoints_tuples_x,
            checkpoints_tuples_y,
            checkpoints_speeds,
            starting_point_landmark,
            finish_point_landmark,
            lane_invasion_checkpoints,
        )


def create_collisions_map(
    experiment_metrics,
    experiment_metrics_filename,
    map_waypoints_tuples_x,
    map_waypoints_tuples_y,
    checkpoints_tuples_x,
    checkpoints_tuples_y,
    checkpoints_speeds,
    starting_point_landmark,
    finish_point_landmark,
    collision_points,
):
    collision_checkpoints_tuples_x = []
    collision_checkpoints_tuples_y = []

    carla_map = experiment_metrics.get("carla_map", "")

    for point in collision_points:
        current_checkpoint = np.array(
            [point["pose.pose.position.x"], point["pose.pose.position.y"]]
        )

        if carla_map in ("Carla/Maps/Town02", "Carla/Maps/Town02_Opt"):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]

        elif carla_map in ("Carla/Maps/Town01", "Carla/Maps/Town01_Opt"):
            checkpoint_x = (
                max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x)
            ) - current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        elif carla_map in (
            "Carla/Maps/Town03",
            "Carla/Maps/Town07",
            "Carla/Maps/Town03_Opt",
            "Carla/Maps/Town07_Opt",
        ):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        elif carla_map in ("Carla/Maps/Town04", "Carla/Maps/Town04_Opt"):
            checkpoint_x = -current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        else:
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]

        collision_checkpoints_tuples_x.append(checkpoint_x)
        collision_checkpoints_tuples_y.append(checkpoint_y)

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    colors = ["#00FF00", "#FF0000", "#000000"]
    ax.scatter(
        map_waypoints_tuples_x,
        map_waypoints_tuples_y,
        s=10,
        c="b",
        marker="s",
        label="Map waypoints",
    )
    plot = ax.scatter(
        checkpoints_tuples_x,
        checkpoints_tuples_y,
        s=10,
        c=checkpoints_speeds,
        cmap="hot_r",
        marker="o",
        label="Experiment waypoints",
        vmin=0,
        vmax=(
            experiment_metrics["max_speed"]
            if experiment_metrics["max_speed"] > 30
            else 30
        ),
    )
    ax.scatter(
        checkpoints_tuples_x[0],
        checkpoints_tuples_y[0],
        s=200,
        marker="o",
        color=colors[0],
        label="Experiment starting point",
    )
    ax.scatter(
        checkpoints_tuples_x[starting_point_landmark],
        checkpoints_tuples_y[starting_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    ax.scatter(
        checkpoints_tuples_x[len(checkpoints_tuples_x) - 1],
        checkpoints_tuples_y[len(checkpoints_tuples_x) - 1],
        s=200,
        marker="o",
        color=colors[1],
        label="Experiment finish point",
    )
    ax.scatter(
        checkpoints_tuples_x[finish_point_landmark],
        checkpoints_tuples_y[finish_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    ax.scatter(
        collision_checkpoints_tuples_x,
        collision_checkpoints_tuples_y,
        s=200,
        marker="o",
        color="y",
        label="Collisions",
    )

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={"size": 20})
    plt.grid(True)
    plt.title(experiment_metrics["experiment_model"] + " Collisions", fontsize=16)
    fig.savefig(experiment_metrics_filename + "_collisions.png", dpi=fig.dpi)


def create_lane_invasions_map(
    experiment_metrics,
    experiment_metrics_filename,
    map_waypoints_tuples_x,
    map_waypoints_tuples_y,
    checkpoints_tuples_x,
    checkpoints_tuples_y,
    checkpoints_speeds,
    starting_point_landmark,
    finish_point_landmark,
    lane_invasion_checkpoints,
):
    lane_invasion_checkpoints_tuples_x = []
    lane_invasion_checkpoints_tuples_y = []

    carla_map = experiment_metrics.get("carla_map", "")

    for point in lane_invasion_checkpoints:
        current_checkpoint = np.array(
            [point["pose.pose.position.x"], point["pose.pose.position.y"]]
        )

        if carla_map in ("Carla/Maps/Town02", "Carla/Maps/Town02_Opt"):
            # Town02: mismo sistema de coordenadas que los checkpoints
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]

        elif carla_map in ("Carla/Maps/Town01", "Carla/Maps/Town01_Opt"):
            checkpoint_x = (
                max(map_waypoints_tuples_x) + min(map_waypoints_tuples_x)
            ) - current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        elif carla_map in (
            "Carla/Maps/Town03",
            "Carla/Maps/Town07",
            "Carla/Maps/Town03_Opt",
            "Carla/Maps/Town07_Opt",
        ):
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        elif carla_map in ("Carla/Maps/Town04", "Carla/Maps/Town04_Opt"):
            checkpoint_x = -current_checkpoint[0]
            checkpoint_y = -current_checkpoint[1]

        else:
            checkpoint_x = current_checkpoint[0]
            checkpoint_y = current_checkpoint[1]

        lane_invasion_checkpoints_tuples_x.append(checkpoint_x)
        lane_invasion_checkpoints_tuples_y.append(checkpoint_y)

    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot()
    colors = ["#00FF00", "#FF0000", "#000000"]
    ax.scatter(
        map_waypoints_tuples_x,
        map_waypoints_tuples_y,
        s=10,
        c="b",
        marker="s",
        label="Map waypoints",
    )
    plot = ax.scatter(
        checkpoints_tuples_x,
        checkpoints_tuples_y,
        s=10,
        c=checkpoints_speeds,
        cmap="hot_r",
        marker="o",
        label="Experiment waypoints",
        vmin=0,
        vmax=(
            experiment_metrics["max_speed"]
            if experiment_metrics["max_speed"] > 30
            else 30
        ),
    )
    ax.scatter(
        checkpoints_tuples_x[0],
        checkpoints_tuples_y[0],
        s=200,
        marker="o",
        color=colors[0],
        label="Experiment starting point",
    )
    ax.scatter(
        checkpoints_tuples_x[starting_point_landmark],
        checkpoints_tuples_y[starting_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    ax.scatter(
        checkpoints_tuples_x[len(checkpoints_tuples_x) - 1],
        checkpoints_tuples_y[len(checkpoints_tuples_x) - 1],
        s=200,
        marker="o",
        color=colors[1],
        label="Experiment finish point",
    )
    ax.scatter(
        checkpoints_tuples_x[finish_point_landmark],
        checkpoints_tuples_y[finish_point_landmark],
        s=100,
        marker="o",
        color=colors[2],
    )
    ax.scatter(
        lane_invasion_checkpoints_tuples_x,
        lane_invasion_checkpoints_tuples_y,
        s=200,
        marker="o",
        color="y",
        label="Lane invasions",
    )

    fig.colorbar(plot, shrink=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", prop={"size": 20})
    plt.grid(True)
    plt.title(experiment_metrics["experiment_model"] + " Lane invasions", fontsize=16)
    fig.savefig(experiment_metrics_filename + "_lane_invasion.png", dpi=fig.dpi)


def get_aggregated_experiments_list(experiments_starting_time):
    current_experiment_folders = []
    root = "./"
    folders = list(os.walk(root))[1:]
    for folder in folders:
        if (
            len(folder[0].split("/")) == 2
            and folder[2]
            and experiments_starting_time < os.stat(folder[0]).st_mtime
            and re.search("\./[0-9]+-[0-9]+", folder[0])
        ):
            current_experiment_folders.append(folder)
    current_experiment_folders.sort()

    dataframes = []
    for folder in current_experiment_folders:
        try:
            r = re.compile(".*\.json")
            json_list = list(filter(r.match, folder[2]))  # Read Note below
            df = pd.read_json(folder[0] + "/" + json_list[0], orient="index").T
            dataframes.append(df)
        except:
            print("Broken experiment: " + folder[0])
            shutil.rmtree(folder[0])

    if not dataframes:
        raise ValueError(
            "No experiment metrics JSON files found. Check if test_suite_manager_carla.py ran correctly."
        )

    result = pd.concat(dataframes)
    result.index = result["timestamp"].values.tolist()
    result.loc[result["collisions"] > 0, "position_deviation_mean"] = float("nan")
    result.loc[result["collisions"] > 0, "effective_completed_distance"] = float("nan")
    result.loc[result["collisions"] > 0, "suddenness_distance_control_commands"] = (
        float("nan")
    )
    result.loc[result["collisions"] > 0, "suddenness_distance_throttle"] = float("nan")
    result.loc[result["collisions"] > 0, "suddenness_distance_steer"] = float("nan")
    result.loc[result["collisions"] > 0, "suddenness_distance_brake_command"] = float(
        "nan"
    )
    result.loc[
        result["collisions"] > 0, "suddenness_distance_control_command_per_km"
    ] = float("nan")
    result.loc[result["collisions"] > 0, "suddenness_distance_throttle_per_km"] = float(
        "nan"
    )
    result.loc[result["collisions"] > 0, "suddenness_distance_steer_per_km"] = float(
        "nan"
    )
    result.loc[result["collisions"] > 0, "suddenness_distance_brake_command_per_km"] = (
        float("nan")
    )
    result.loc[result["collisions"] > 0, "completed_distance"] = float("nan")
    result.loc[result["collisions"] > 0, "average_speed"] = float("nan")
    result.loc[result["collisions"] > 0, "position_deviation_mean"] = float("nan")
    result.loc[result["collisions"] > 0, "position_deviation_mean_per_km"] = float(
        "nan"
    )
    result.loc[result["collisions"] > 0, "suddenness_distance_speed"] = float("nan")
    result.loc[result["collisions"] > 0, "suddenness_distance_speed_per_km"] = float(
        "nan"
    )

    return result


def get_maps_colors():
    maps_colors = {
        "Carla/Maps/Town01": "red",
        "Carla/Maps/Town02": "green",
        "Carla/Maps/Town03": "blue",
        "Carla/Maps/Town04": "grey",
        "Carla/Maps/Town05": "black",
        "Carla/Maps/Town06": "pink",
        "Carla/Maps/Town07": "orange",
    }
    return maps_colors


def get_color_handles():
    red_patch = mpatches.Patch(color="red", label="Map01")
    green_patch = mpatches.Patch(color="green", label="Map02")
    blue_patch = mpatches.Patch(color="blue", label="Map03")
    grey_patch = mpatches.Patch(color="grey", label="Map04")
    black_patch = mpatches.Patch(color="black", label="Map05")
    pink_patch = mpatches.Patch(color="pink", label="Map06")
    orange_patch = mpatches.Patch(color="orange", label="Map07")
    color_handles = [
        red_patch,
        green_patch,
        blue_patch,
        grey_patch,
        black_patch,
        pink_patch,
        orange_patch,
    ]

    return color_handles


def get_all_experiments_aggregated_metrics(
    result, experiments_starting_time_str, experiments_metrics_and_titles
):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    colors = []
    for i in result["carla_map"]:
        colors.append(maps_colors[i])

    for experiment_metric_and_title in experiments_metrics_and_titles:
        fig = plt.figure(figsize=(40, 10))
        result[experiment_metric_and_title["metric"]].plot.bar(color=colors)
        plt.title(experiment_metric_and_title["title"])
        fig.tight_layout()
        plt.xticks(rotation=90)
        plt.legend(handles=color_handles)
        plt.savefig(
            experiments_starting_time_str
            + "/"
            + experiment_metric_and_title["metric"]
            + ".png"
        )
        plt.close()


def get_per_model_aggregated_metrics(
    result, experiments_starting_time_str, experiments_metrics_and_titles
):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    unique_experiment_models = result["experiment_model"].unique()

    for unique_experiment_model in unique_experiment_models:
        unique_model_experiments = result.loc[
            result["experiment_model"].eq(unique_experiment_model)
        ]
        colors = []
        for i in unique_model_experiments["carla_map"]:
            colors.append(maps_colors[i])

        for experiment_metric_and_title in experiments_metrics_and_titles:
            fig = plt.figure(figsize=(40, 10))
            unique_model_experiments[experiment_metric_and_title["metric"]].plot.bar(
                color=colors
            )
            plt.title(
                experiment_metric_and_title["title"]
                + " with "
                + unique_experiment_model
            )
            fig.tight_layout()
            plt.xticks(rotation=90)
            plt.legend(handles=color_handles)
            if len(unique_experiment_model.split("/")) > 1:
                unique_experiment_model = unique_experiment_model.split("/")[-1]
            plt.savefig(
                experiments_starting_time_str
                + "/"
                + unique_experiment_model
                + "_"
                + experiment_metric_and_title["metric"]
                + ".png"
            )
            plt.close()


def get_all_experiments_aggregated_metrics_boxplot(
    result, experiments_starting_time_str, experiments_metrics_and_titles
):
    maps_colors = get_maps_colors()
    color_handles = get_color_handles()
    for experiment_metric_and_title in experiments_metrics_and_titles:
        fig = plt.figure(figsize=(40, 10))
        dataframes = []
        max_value = 0
        for x, model_name in enumerate(result["experiment_model"].unique()):
            for y, carla_map in enumerate(result["carla_map"].unique()):
                com_dict = {
                    "model_name": model_name,
                    model_name
                    + "-"
                    + carla_map: result.loc[
                        (result["experiment_model"] == model_name)
                        & (result["carla_map"] == carla_map)
                    ][experiment_metric_and_title["metric"]].tolist(),
                    "carla_map": carla_map,
                }
                df = pd.DataFrame(data=com_dict)
                dataframes.append(df)
                if df[model_name + "-" + carla_map].max() > max_value:
                    max_value = df[model_name + "-" + carla_map].max()

        full_list = []
        colors = []
        for carla_map in result["carla_map"].unique().tolist():
            for experiment_model in result["experiment_model"].unique().tolist():
                full_list.append(experiment_model + "-" + carla_map)
                colors.append(maps_colors[carla_map])

        result_by_experiment_model = pd.concat(dataframes)
        ax, props = result_by_experiment_model.boxplot(
            column=full_list,
            showfliers=True,
            sym="k.",
            return_type="both",
            patch_artist=True,
        )

        plt.title(experiment_metric_and_title["title"] + " boxplot")

        for patch, color in zip(props["boxes"], colors):
            patch.set_facecolor(color)

        plt.legend(handles=color_handles)

        plt.xticks(rotation=90)
        if max_value > 0:
            plt.ylim(0, max_value + max_value * 0.1)
        plt.savefig(
            experiments_starting_time_str
            + "/"
            + experiment_metric_and_title["metric"]
            + "_boxplot.png",
            bbox_inches="tight",
        )
        plt.close()


def get_distance_other_vehicle(experiment_metrics, checkpoints, checkpoints_2):
    dangerous_distance = 0
    close_distance = 0
    medium_distance = 0
    great_distance = 0
    total_distance = 0

    dangerous_distance_pct_km = 0
    close_distance_pct_km = 0
    medium_distance_pct_km = 0
    great_distance_pct_km = 0
    total_distance_pct_km = 0

    for i, (point, point_2) in enumerate(zip(checkpoints, checkpoints_2)):
        current_checkpoint = np.array(
            [point["pose.pose.position.x"], point["pose.pose.position.y"]]
        )
        current_checkpoint_2 = np.array(
            [point_2["pose.pose.position.x"], point_2["pose.pose.position.y"]]
        )

        if i != 0:
            distance_front = np.linalg.norm(current_checkpoint - current_checkpoint_2)
            distance = np.linalg.norm(previous_point - current_checkpoint)

            # Analyzing
            if 20 < distance_front < 50:
                great_distance += distance
                total_distance += distance
            elif 15 < distance_front <= 20:
                medium_distance += distance
                total_distance += distance
            elif 6 < distance_front <= 15:
                close_distance += distance
                total_distance += distance
            elif distance_front <= 6:
                dangerous_distance += distance
                total_distance += distance

        previous_point = current_checkpoint

    experiment_metrics["dangerous_distance_km"] = dangerous_distance
    experiment_metrics["close_distance_km"] = close_distance
    experiment_metrics["medium_distance_km"] = medium_distance
    experiment_metrics["great_distance_km"] = great_distance
    experiment_metrics["total_distance_to_front_car"] = total_distance

    experiment_metrics["dangerous_distance_pct_km"] = (
        total_distance and dangerous_distance / total_distance or 0
    ) * 100
    experiment_metrics["close_distance_pct_km"] = (
        total_distance and close_distance / total_distance or 0
    ) * 100
    experiment_metrics["medium_distance_pct_km"] = (
        total_distance and medium_distance / total_distance or 0
    ) * 100
    experiment_metrics["great_distance_pct_km"] = (
        total_distance and great_distance / total_distance or 0
    ) * 100

    return experiment_metrics


# Python API
def _na_json(v):
    try:
        if v is None:
            return "N/A"
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return "N/A"
        return float(v) if isinstance(v, (int, float)) else v
    except Exception:
        return "N/A"


def build_json_summary(experiment_metrics: dict, extras: dict = None) -> dict:
    """
    Toma el dict completo 'experiment_metrics' que ya construye get_metrics(...)
    y devuelve un resumen estable para JSON.
    """
    keys = [
        "experiment_total_real_time",
        "experiment_total_simulated_time",
        "completed_distance",
        "effective_completed_distance",
        "percentage_completed",
        "completed_laps",
        "average_speed",
        "max_speed",
        "min_speed",
        "position_deviation_mean",
        "position_deviation_total_err",
        "position_deviation_mean_per_km",
        "collisions",
        "collisions_per_km",
        "lane_invasions",
        "lane_invasions_per_km",
        "suddenness_distance_control_commands",
        "suddenness_distance_throttle",
        "suddenness_distance_steer",
        "suddenness_distance_brake_command",
        "suddenness_distance_speed",
        "suddenness_distance_control_command_per_km",
        "suddenness_distance_throttle_per_km",
        "suddenness_distance_steer_per_km",
        "suddenness_distance_brake_command_per_km",
        "suddenness_distance_speed_per_km",
        "experiment_model",
        "carla_map",
        "starting_point",
        "starting_point_map",
    ]

    out = {}
    for k in keys:
        if k in experiment_metrics:
            out[k] = _na_json(experiment_metrics[k])

    # extras opcionales (modelo, ruta, timestamp, etc.)
    if extras:
        for k, v in extras.items():
            out[k] = _na_json(v)

    return out


def save_json_summary(json_path: str, summary: dict):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)


# --- BACKEND PYTHON API (CSV) ---


def get_metrics_python_api(
    experiment_metrics, csv_path, map_waypoints, experiment_metrics_filename, config
):
    """
    Lee el CSV que genera ControllerCarla._csv_open/_csv_append y devuelve experiment_metrics
    con un set básico de métricas (distancia, velocidades, súbitas, colisiones, invasiones).
    Generación de mapas se puede añadir luego si lo necesitas igual que en el flujo ROS.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[CSV metrics] No pude leer {csv_path}: {e}")
        return {}

    # Asegura columnas esperadas
    for c in [
        "x",
        "y",
        "speed_mps",
        "throttle",
        "steer",
        "brake",
        "collision_impulse",
        "collision_with",
        "lane_invasion",
        "time_stamp",
        "frame",
    ]:
        if c not in df.columns:
            df[c] = 0 if c not in ("collision_with", "lane_invasion") else ""

    # Tiempo simulado (si existe time_stamp usamos extremos)
    try:
        experiment_metrics["experiment_total_simulated_time"] = float(
            df["time_stamp"].iloc[-1] - df["time_stamp"].iloc[0]
        )
    except Exception:
        experiment_metrics["experiment_total_simulated_time"] = float(
            len(df) / 20.0
        )  # aprox 20 Hz

    # Distancia recorrida (en el plano XY)
    if len(df) >= 2:
        p = df[["x", "y"]].to_numpy(dtype=float)
        d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1)).sum()
        experiment_metrics["completed_distance"] = float(d)
    else:
        experiment_metrics["completed_distance"] = 0.0

    # Velocidades
    v_mps = df["speed_mps"].astype(float).to_numpy()
    if len(v_mps) > 0:
        experiment_metrics["average_speed"] = float(v_mps.mean() * 3.6)  # km/h
        experiment_metrics["max_speed"] = float(v_mps.max() * 3.6)
        experiment_metrics["min_speed"] = float(v_mps.min() * 3.6)
    else:
        experiment_metrics["average_speed"] = experiment_metrics["max_speed"] = (
            experiment_metrics["min_speed"]
        ) = 0.0

    def l2_sudden(series):
        s = series.astype(float).to_numpy()
        if len(s) < 2:
            return 0.0
        return float(np.mean(np.abs(s[1:] - s[:-1])))

    experiment_metrics["suddenness_distance_throttle"] = l2_sudden(df["throttle"])
    experiment_metrics["suddenness_distance_steer"] = l2_sudden(df["steer"])
    experiment_metrics["suddenness_distance_brake_command"] = l2_sudden(df["brake"])
    experiment_metrics["suddenness_distance_control_commands"] = float(
        np.mean(
            np.sqrt(
                (df["throttle"].diff().fillna(0).astype(float)) ** 2
                + (df["steer"].diff().fillna(0).astype(float)) ** 2
                + (df["brake"].diff().fillna(0).astype(float)) ** 2
            )
        )
    )

    # Colisiones 
    col_thr = 0.1
    collisions_idx = df.index[df["collision_impulse"].astype(float) > col_thr]
    experiment_metrics["collisions"] = int(len(collisions_idx))
    experiment_metrics["collision_actor_ids"] = [
        str(x) for x in df.loc[collisions_idx, "collision_with"].tolist()
    ]

    # Invasiones de carril (cadena no vacía)
    lane_idx = df.index[df["lane_invasion"].astype(str).str.strip() != ""]
    experiment_metrics["lane_invasions"] = int(len(lane_idx))

    # Per-km (evita división por cero)
    km = max(experiment_metrics["completed_distance"] / 1000.0, 1e-9)
    experiment_metrics["collisions_per_km"] = experiment_metrics["collisions"] / km
    experiment_metrics["lane_invasions_per_km"] = (
        experiment_metrics["lane_invasions"] / km
    )
    experiment_metrics["suddenness_distance_control_command_per_km"] = (
        experiment_metrics["suddenness_distance_control_commands"] / km
    )
    experiment_metrics["suddenness_distance_throttle_per_km"] = (
        experiment_metrics["suddenness_distance_throttle"] / km
    )
    experiment_metrics["suddenness_distance_steer_per_km"] = (
        experiment_metrics["suddenness_distance_steer"] / km
    )
    experiment_metrics["suddenness_distance_brake_command_per_km"] = (
        experiment_metrics["suddenness_distance_brake_command"] / km
    )


    experiment_metrics.setdefault("percentage_completed", 0.0)
    experiment_metrics.setdefault(
        "effective_completed_distance", experiment_metrics["completed_distance"]
    )
    experiment_metrics.setdefault("position_deviation_mean", 0.0)
    experiment_metrics.setdefault("position_deviation_total_err", 0.0)
    experiment_metrics.setdefault("position_deviation_mean_per_km", 0.0)
    experiment_metrics.setdefault("completed_laps", 0)

    # Metrics plot (*.png file)
    try:
        checkpoints = [
            {"pose.pose.position.x": float(x), "pose.pose.position.y": float(y)}
            for x, y in df[["x", "y"]].to_numpy()
        ]

        speedometer_points = [
            {"data": float(v)} for v in df["speed_mps"].astype(float).to_numpy()
        ]

        col_thr = 0.1
        collisions_idx = df.index[
            df["collision_impulse"].astype(float) > col_thr
        ].tolist()
        collisions_checkpoints = [
            {
                "pose.pose.position.x": float(df.at[i, "x"]),
                "pose.pose.position.y": float(df.at[i, "y"]),
            }
            for i in collisions_idx
        ]

        lane_idx = df.index[df["lane_invasion"].astype(str).str.strip() != ""].tolist()
        lane_invasion_checkpoints = [
            {
                "pose.pose.position.x": float(df.at[i, "x"]),
                "pose.pose.position.y": float(df.at[i, "y"]),
            }
            for i in lane_idx
        ]

        experiment_metrics = get_position_deviation_and_effective_completed_distance(
            experiment_metrics=experiment_metrics,
            checkpoints=checkpoints,
            map_waypoints=map_waypoints,
            experiment_metrics_filename=experiment_metrics_filename,
            speedometer=speedometer_points,
            collision_points=collisions_checkpoints,
            lane_invasion_checkpoints=lane_invasion_checkpoints,
        )

    except Exception as e:
        print(f"[CSV metrics] Falló el postproceso de mapa/errores/PNG: {e}")

    return experiment_metrics
