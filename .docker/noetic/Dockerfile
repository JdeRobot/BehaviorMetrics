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
    #pip3 install -r requirements.txt && \
    pip3 install --no-deps carla_birdeye_view==1.1.1 && \
    pyrcc5 -o behavior_metrics/ui/gui/resources/resources.py \
	behavior_metrics/ui/gui/resources/resources.qrc

# Installing RoboticsInfrastructure/CustomRobots
#RUN git clone -b noetic-devel https://github.com/JdeRobot/RoboticsInfrastructure && \
#    cd RoboticsInfrastructure/CustomRobots/f1 && mkdir build && cd build && \
#    /bin/bash -c "source /opt/ros/noetic/setup.bash; \
#		  cmake .. && make && make install;" && \
#    echo "source /opt/jderobot/share/jderobot/gazebo/assets-setup.sh" >> ~/.bashrc

RUN echo 'alias jl="DISPLAY=:0 jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.bashrc && \
    echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    git clone https://github.com/fmtlib/fmt.git && \
        cd fmt && git checkout 5.3.0 && mkdir build && cd build && \
        cmake ../ && make && make install && cd && \
    git clone https://github.com/strasdat/Sophus && \
	cd Sophus && mkdir build && cd build && \
	cmake ../ && make && make install

RUN  sudo apt-get -y install jupyter && \
    pip3 install --ignore-installed jupyterlab_server && \
    pip3 uninstall -y jupyterlab && pip3 install jupyterlab && \
    pip3 install Xlib

COPY ./vnc_startup.sh /

WORKDIR /root

ENTRYPOINT ["../vnc_startup.sh"]
CMD ["bash"]
