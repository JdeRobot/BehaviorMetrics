#!/bin/sh
cd gym_gazebo/envs/installation
rm -rf catkin_ws
bash setup_noetic.bash
bash formula1_laser_setup.bash
cd ~/BehaviorStudio/
