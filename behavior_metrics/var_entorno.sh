#!/bin/bash

# Variables de entorno para CARLA y ROS 2
export ROS_VERSION=2
export CARLA_ROOT=$HOME/carla_simulator/
export OBJECT_PATH=$HOME/BehaviorMetrics/behavior_metrics/configs/CARLA/CARLA_launch_files/CARLA_object_files/parked_car_objects.json
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg

echo "Environment variables set up successfully!"

