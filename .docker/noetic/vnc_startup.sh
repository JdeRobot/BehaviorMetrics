#!/bin/sh
export DISPLAY=:0
Xvfb :0 -screen 0 1920x1080x24 &
icewm-session &
x11vnc -display :0 -passwd jderobot -forever -noxdamage &
