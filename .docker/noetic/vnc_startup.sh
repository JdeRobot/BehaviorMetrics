#!/bin/bash

# Make sure there aren't ohther displays running before launch vnc.
rm -rfv ~/tmp
mkdir ~/tmp
chmod 777 ~/tmp
export DISPLAY=:0
Xvfb :0 -screen 0 1920x1080x24 &
icewm-session &
x11vnc -display :0 -passwd mishra -forever -noxdamage &

# Source noetic
source ~/.bashrc

# Start jupyter lab
DISPLAY=:0 jupyter lab --no-browser --ip 0.0.0.0 --port 8888 --allow-root &

exec "$@"
