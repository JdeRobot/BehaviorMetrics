FROM jderobot/ubuntu:ros-noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654

RUN apt-get update && apt-get install -y \
        cmake \
        icewm \
        git \
        zip \
        qtcreator \
        psmisc \
        build-essential \
        genromfs \
        ninja-build \
        exiftool \
        python3-setuptools \
        python3-pip \
        python3-dev \
        python-is-python3 \
        python3-rviz \
        python3-opengl \
        python3-catkin-tools \
        python3-osrf-pycommon \
        python3-rosdep \
        tmux \
        vim \
        x11vnc \
        software-properties-common \
        xvfb && \
    pip3 install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN  add-apt-repository ppa:deadsnakes/ppa && \
      apt-get update && \
      apt-get install -y apt-utils

# Remove possible duplicated packages
RUN  pip install PyQt5==5.14.1 --upgrade --ignore-installed && \
     pip install PyYAML==5.4  --upgrade --ignore-installed && \
     apt-get -y install python3-tk

# Installing Behavior Metrics
RUN cd /root/ && \
    git clone -b noetic-devel https://github.com/JdeRobot/BehaviorMetrics && \
    cd BehaviorMetrics && \
    pip3 install -r requirements.txt && \
    pyrcc5 -o behavior_metrics/ui/gui/resources/resources.py \
	behavior_metrics/ui/gui/resources/resources.qrc

# Installing CustomRobots
RUN git clone -b noetic-devel https://github.com/JdeRobot/CustomRobots && \
    cd CustomRobots/f1 && mkdir build && cd build && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; \
		  cmake .. && make && make install;" && \
    echo "source /opt/jderobot/share/jderobot/gazebo/assets-setup.sh" >> ~/.bashrc

RUN echo 'alias jl="DISPLAY=:0 jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.bashrc && \
    echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    git clone https://github.com/fmtlib/fmt.git && \
        cd fmt && git checkout 5.3.0 && mkdir build && cd build && \
        cmake ../ && make && make install && cd && \
    git clone https://github.com/strasdat/Sophus && \
	cd Sophus && mkdir build && cd build && \
	cmake ../ && make && make install

# Installing Gym-Gazebo
RUN apt-get update && apt-get install -y \
	libbluetooth-dev \
	libcwiid-dev \
	libftdi-dev \
	libspnav-dev \
	libsdl-dev \
	libsdl-image1.2-dev \
 	libusb-dev \
	ros-noetic-octomap-msgs        \
	ros-noetic-geodesy             \
	ros-noetic-octomap-ros         \
	ros-noetic-control-toolbox     \
	ros-noetic-pluginlib	       \
	ros-noetic-trajectory-msgs     \
	ros-noetic-control-msgs	       \
	ros-noetic-std-srvs 	       \
	ros-noetic-nodelet	       \
	ros-noetic-urdf		       \
	ros-noetic-rviz		       \
	ros-noetic-kdl-conversions     \
	ros-noetic-eigen-conversions   \
	ros-noetic-tf2-sensor-msgs     \
	ros-noetic-pcl-ros \
	ros-noetic-navigation \
    ros-noetic-mavros \
    ros-noetic-mavros-extras && \
    cd root/BehaviorMetrics/gym-gazebo/ && \
    pip3 install -e . && \
    pip3 install rospkg --upgrade && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

RUN cd /root/ && \
    wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && \
    chmod +x install_geographiclib_datasets.sh && \
    sudo ./install_geographiclib_datasets.sh && \
    rm ./install_geographiclib_datasets.sh

RUN cd /root/ && \
    mkdir /root/repos && cd /root/repos && \
    git clone --recursive https://github.com/PX4/PX4-Autopilot.git -b v1.11.3 && \
    cd /root/repos/PX4-Autopilot/Tools/setup/ && \
    bash ubuntu.sh --no-nuttx --no-sim-tools

RUN apt-get install -y \
    libgstreamer1.0-dev \
    gstreamer1.0-plugins-bad

RUN cd /root/ && \
    cd /root/repos/PX4-Autopilot && \
    DONT_RUN=1 make px4_sitl gazebo

RUN echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/repos/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> /root/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/repos/PX4-Autopilot/Tools/sitl_gazebo/models' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/repos/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> /root/.bashrc && \
    echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/repos/PX4-Autopilot' >> /root/.bashrc && \
    echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/repos/PX4-Autopilot/Tools/sitl_gazebo' >> /root/.bashrc && \
    /bin/bash -c ". /opt/ros/noetic/setup.bash; source ~/.bashrc;"

## Install jderobot_drones source
RUN rm -rf /catkin_ws
RUN mkdir -p /catkin_ws/src
RUN cd /catkin_ws && catkin init
RUN echo 'export ROS_WORKSPACE=/catkin_ws' >> ~/.bashrc

RUN git clone https://github.com/JdeRobot/drones.git && cd drones && git checkout noetic-devel
RUN cd /catkin_ws/src && ln -s /drones/drone_wrapper .
RUN cd /catkin_ws/src && ln -s /drones/drone_assets .
RUN cd /catkin_ws/src && ln -s /drones/rqt_drone_teleop .

RUN /bin/bash -c '. /opt/ros/noetic/setup.bash' && cd /catkin_ws && \
    rm /etc/ros/rosdep/sources.list.d/20-default.list && \
    sudo rosdep init 

# RUN rosdep update && rosdep install --from-paths . --ignore-src --rosdistro noetic -y

RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; cd /catkin_ws; catkin build'

RUN cd /drones/drone_circuit_assets && mkdir build && cd build && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; \
		  cmake .. && make && make install;"

RUN cd /catkin_ws/ && \
    echo 'source '$PWD'/devel/setup.bash' >> /root/.bashrc  && \
    echo 'export GAZEBO_RESOURCE_PATH=${GAZEBO_RESOURCE_PATH}:/usr/share/gazebo-11' >> /root/.bashrc  && \
    echo 'export GAZEBO_MODEL_PATH=${GAZEBO_MODEL_PATH}:/drones/drone_assets/models' >> /root/.bashrc  && \
    echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/repos/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> /root/.bashrc && \
    echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/repos/PX4-Autopilot/Tools/sitl_gazebo/models' >> /root/.bashrc && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/repos/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> /root/.bashrc && \
    echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/repos/PX4-Autopilot' >> /root/.bashrc && \
    echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/repos/PX4-Autopilot/Tools/sitl_gazebo' >> /root/.bashrc && \
    /bin/bash -c ". ~/.bashrc;"

COPY ./vnc_startup.sh /

WORKDIR /root

ENTRYPOINT ["../vnc_startup.sh"]
CMD ["bash"]
