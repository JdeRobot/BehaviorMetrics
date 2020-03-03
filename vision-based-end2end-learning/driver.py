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
#  Authors:
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>
#       Franciso Perez Salgado <f.perez475@gmail.com>
#       Ignacio Arranz Agueda <n.arranz.agueda@gmail.com>
#  Based on:
#       Follow line code: https://github.com/JdeRobot/Academy/tree/master/exercises/follow_line
#       and @naxvm code: https://github.com/JdeRobot/dl-objectdetector
#


# General imports
import sys
import yaml
import importlib
import logging

# Practice imports
from gui.GUI import MainWindow
from gui.threadGUI import ThreadGUI
from MyAlgorithm import MyAlgorithm
from autopilot import AutoPilot
from PyQt5.QtWidgets import QApplication
from interfaces.camera import ListenerCamera
from interfaces.motors import PublisherMotors
from net.threadNetwork import ThreadNetwork
from network_configurator import NetworkConfigurator


class Colors:
    """
    Colors defined for improve the prints in each Stage
    """
    DEBUG = '\033[1;36;1m'
    OKCYAN = '\033[96m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def __init__(self):
        pass


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

    logging.basicConfig(format='[%(levelname)s]{} %(message)s{}'.format(Colors.OKBLUE, Colors.ENDC), level=logging.INFO)

    cfg = readConfig()
    configurator = NetworkConfigurator(cfg)
    network = configurator.create_network()

    camera = ListenerCamera("/F1ROS/cameraL/image_raw")
    motors = PublisherMotors("/F1ROS/cmd_vel", 4, 0.3, 0, 0)    

    network.setCamera(camera)
    t_network = ThreadNetwork(network)
    t_network.start()

    algorithm = MyAlgorithm(camera, motors, network)
    autopilot = AutoPilot(camera, motors)
    # algorithm = MyAlgorithm(None, None, None)

    app = QApplication(sys.argv)
    myGUI = MainWindow()
    myGUI.setCamera(camera)
    myGUI.setMotors(motors)
    myGUI.setAlgorithm(algorithm)
    myGUI.setAutopilot(autopilot)
    myGUI.setThreadConnector(t_network)
    myGUI.show()

    t2 = ThreadGUI(myGUI)
    t2.daemon=True
    t2.start()

    sys.exit(app.exec_())
