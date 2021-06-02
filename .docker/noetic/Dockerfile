FROM jderobot/ubuntu:ros-noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654

RUN apt-get update && apt-get install -y \
        cmake \
        icewm \
        git \
        psmisc \
        python3-setuptools \
        python3-pip \
        python3-rviz \
        python3-opengl \
        pyqt5-dev-tools \
        tmux \
        vim \
        x11vnc \
        xvfb && \
    pip3 install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
	software-properties-common && \
	add-apt-repository ppa:deadsnakes/ppa && \
	apt-get update && \
	apt-get install -y python3.7 python3.7-dev && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# Installing Behavior Metrics
RUN cd /root/ && \
    git clone -b noetic-devel https://github.com/JdeRobot/BehaviorMetrics && \
    cd BehaviorMetrics && \
    pip3 install -r requirements.txt && \
    pip3 -V && python3 -V && \
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
	ros-noetic-navigation && \
    cd root/BehaviorMetrics/gym-gazebo/ && \
    pip3 install -e . && \	
#    ./load_env.sh && \
    apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

COPY ./vnc_startup.sh /

WORKDIR /root

ENTRYPOINT ["../vnc_startup.sh"]
CMD ["bash"]
