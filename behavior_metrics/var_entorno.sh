#!/bin/bash

# Environment variables for CARLA in Python API mode (No ROS)
unset ROS_VERSION
export ROS_VERSION=None
export CARLA_ROOT=$HOME/code/CarlaSim
export OBJECT_PATH=$HOME/code/BehaviorMetrics/behavior_metrics/configs/CARLA/CARLA_launch_files/CARLA_object_files/parked_car_objects.json
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla

echo "Environment variables set up for CARLA Python API (ROS_VERSION=None, CARLA_ROOT=$CARLA_ROOT)"
