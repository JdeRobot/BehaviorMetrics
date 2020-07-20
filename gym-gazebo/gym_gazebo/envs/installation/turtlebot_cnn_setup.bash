#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../assets/models'," -i ~/.bashrc'
fi

#Load turtlebot variables. Temporal solution
chmod +x catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em
bash catkin_ws/src/turtlebot_simulator/turtlebot_gazebo/env-hooks/25.turtlebot-gazebo.sh.em

#add turtlebot launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT="`pwd`/../assets/worlds/circuit.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCUIT=[^;]*,'GYM_GAZEBO_WORLD_CIRCUIT=`pwd`/../assets/worlds/circuit.world'," -i ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT2" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT2="`pwd`/../assets/worlds/circuit2.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCUIT2=[^;]*,'GYM_GAZEBO_WORLD_CIRCUIT2=`pwd`/../assets/worlds/circuit2.world'," -i ~/.bashrc'
fi
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT2C" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT2C="`pwd`/../assets/worlds/circuit2c.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCUIT2C=[^;]*,'GYM_GAZEBO_WORLD_CIRCUIT2C=`pwd`/../assets/worlds/circuit2c.world'," -i ~/.bashrc'
fi

exec bash # reload bash

