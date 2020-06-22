#!/bin/sh

# Make sure there aren't ohther displays running before launch vnc.
rm -rfv ~/tmp
mkdir ~/tmp
chmod 777 ~/tmp
export DISPLAY=:0
Xvfb :0 -screen 0 1920x1080x24 &
icewm-session &
x11vnc -display :0 -passwd jderobot -forever -noxdamage &


# Source noetic
source ~/.bashrc

exec "$@"
