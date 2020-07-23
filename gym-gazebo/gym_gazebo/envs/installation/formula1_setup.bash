#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

#Load turtlebot variables. Temporal solution
#chmod +x catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em
#bash catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em

#add Formula 1 launch environment variable

#if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT_F1" ]; then
#  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT="`pwd`/../assets/worlds/circuit.world >> ~/.bashrc'
#else
#  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCUIT=[^;]*,'GYM_GAZEBO_WORLD_CIRCUIT=`pwd`/../assets/worlds/f1_1_simplecircuit.world'," -i ~/.bashrc'
#fi
bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT_F1="`pwd`/../assets/worlds/f1_1_simplecircuit.world >> ~/.bashrc'
#export GYM_GAZEBO_WORLD_CIRCUIT_F1=`pwd`/../assets/worlds/f1_1_simplecircuit.world
echo 'Formula 1 env variables loaded succesfully'
 
exec bash # reload bash

