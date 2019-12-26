#!/usr/bin/python3
#
#  Copyright (C) 1997-2018 JdeRobot Developers Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/.
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>
#  Based on:
#       Follow line code: https://github.com/JdeRobot/Academy/tree/master/exercises/follow_line
#       and @naxvm code: https://github.com/JdeRobot/dl-objectdetector
#


# General imports
import sys
import yaml
import importlib

# Practice imports
from gui.GUI import MainWindow
from gui.threadGUI import ThreadGUI
from MyAlgorithm import MyAlgorithm
from PyQt5.QtWidgets import QApplication
from interfaces.camera import ListenerCamera
from interfaces.motors import PublisherMotors
from net.threadNetwork import ThreadNetwork
from network_configurator import NetworkConfiurator


def readConfig():
    try:
        with open(sys.argv[1], 'r') as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        raise SystemExit('Error: Cannot read/parse YML file. Check YAML syntax.')
    except:
        raise SystemExit('\n\tUsage: python2 driver.py driver.yml\n')


if __name__ == "__main__":

    cfg = readConfig()
    configurator = NetworkConfiurator(cfg)
    network = configurator.create_network()

    camera = ListenerCamera("/F1ROS/cameraL/image_raw")
    motors = PublisherMotors("/F1ROS/cmd_vel", 4, 0.3, 0, 0)    

    network.setCamera(camera)
    t_network = ThreadNetwork(network)
    t_network.start()

    algorithm = MyAlgorithm(camera, motors, network)

    app = QApplication(sys.argv)
    myGUI = MainWindow()
    myGUI.setCamera(camera)
    myGUI.setMotors(motors)
    myGUI.setAlgorithm(algorithm)
    myGUI.show()

    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()

    sys.exit(app.exec_())
