FROM uddua/jderobot-base:noetic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	cmake \
	icewm \
	git \
	psmisc \
	python3-pip \
 	python3-setuptools \
	python3-opengl \ 
        python3-pyqt5 \
	python3-pyqt5.qtsvg \
	python3-pyqt5.qsci \
	pyqt5-dev-tools \
	tmux \
	x11vnc \
	xvfb && \
    pip3 install --upgrade pip && \
    pip3 install jupyterlab && \
    rm -rf /var/lib/apt/lists/*

# Installing behavior studio
RUN cd /root/ && \
    git clone -b noetic-devel https://github.com/dcharrezt/BehaviorStudio && \
    cd BehaviorStudio && \
    pip3 install -r requirements.txt && \
    pyrcc5 -o behavior_studio/ui/gui/resources/resources.py \
	behavior_studio/ui/gui/resources/resources.qrc

# Installing Jderobot-assets
RUN git clone https://github.com/JdeRobot/assets && \
    cd assets/jderobot_assets && mkdir build && cd build && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash; \
    		  cmake .. && make && make install;" && \
    echo "source /opt/jderobot/share/jderobot/gazebo/assets-setup.sh" >> ~/.bashrc

RUN echo 'alias jl="DISPLAY=:0 jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.bashrc && \
    echo "alias killgazebogym='killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient'" >> ~/.bashrc && \
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

COPY ./vnc_startup.sh /

WORKDIR /root

ENTRYPOINT ["../vnc_startup.sh"]
CMD ["bash"]
