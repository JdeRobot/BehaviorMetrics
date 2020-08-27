#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT_F1="`pwd`/../assets/worlds/f1_1_simplecircuit.world >> ~/.bashrc'
bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT_F1_LASER="`pwd`/../assets/worlds/f1_1_simplecircuit_laser.world >> ~/.bashrc'
#export GYM_GAZEBO_WORLD_CIRCUIT_F1=`pwd`/../assets/worlds/f1_1_simplecircuit.world
echo 'Formula 1 env variables loaded succesfully'
 
exec bash # reload bash

