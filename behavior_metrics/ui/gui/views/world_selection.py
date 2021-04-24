#!/usr/bin/env python
""" This module contains the world selection view.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QSequentialAnimationGroup, QSize
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QWidget, QLabel, QHBoxLayout, QVBoxLayout, QGraphicsOpacityEffect, QFrame, QListWidget,
                             QAbstractItemView, QListWidgetItem)

from ui.gui.views.logo import Logo
from ui.gui.views.models3d import View3D

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

resources_path = str(Path(__file__).parent.parent) + '/resources/'


class InfoLabel(QLabel):
    """This class extends the default functionality of the QLabel adding a tooltip to it"""

    def __init__(self, description, parent=None):
        """Constructor of the class

        Arguments:
            description {str} -- Tooltip text
            parent {ui.gui.views.main_view.MainView} -- Parent of the widget

        """
        QLabel.__init__(self, parent)
        self.description = description

        self.setFixedSize(60, 60)
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))
        self.setMouseTracking(True)
        self.setStyleSheet("""
                                QToolTip {
                                    background-color: rgb(51,51,51);
                                    color: white;
                                    border: black solid 1px;
                                    font-size: 20px;
                                }""")
        self.setToolTip(self.description)

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(60, 60, Qt.KeepAspectRatio))

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))


# class WorldLabel(QLabel):

#     def __init__(self, parent=None):
#         QLabel.__init__(self, parent)
#         self.setMouseTracking(True)
#         self.parent = parent
#         self.setStyleSheet('color: white')

#     def enterEvent(self, event):
#         self.setStyleSheet('color: yellow')

#     def leaveEvent(self, event):
#         self.setStyleSheet('color: white')

#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self.parent.parent.generate_launch_file(self.parent.world['world'])


class ClickableLabel(QLabel):
    """Class that extends the functionality of the QLabel adding mouse events"""

    clicked = pyqtSignal()

    in_label = False
    SLOW_DURATION = 1500
    FAST_DURATION = 500

    def __init__(self, parent=None):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views.main_view.world_selection.WorldSelection} -- Parent of this widget (default: {None})
        """
        QLabel.__init__(self, parent)
        self.start_animation(self.SLOW_DURATION)

    def start_animation(self, duration):
        """Start a fadein-fadeout animation

        Arguments:
            duration {int} -- Duration of millisecondns of each fadein-fadeout cycle of the animation.
        """
        self.effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.effect)

        self.animation1 = QPropertyAnimation(self.effect, b"opacity")
        self.animation1.setDuration(duration)
        self.animation1.setStartValue(1)
        self.animation1.setEndValue(0)

        self.animation2 = QPropertyAnimation(self.effect, b"opacity")
        self.animation2.setDuration(duration)
        self.animation2.setStartValue(0)
        self.animation2.setEndValue(1)

        self.ga = QSequentialAnimationGroup()
        self.ga.addAnimation(self.animation1)
        self.ga.addAnimation(self.animation2)
        self.ga.setLoopCount(-1)
        self.ga.start()

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        self.ga.stop()
        self.start_animation(self.FAST_DURATION)

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.ga.stop()
        self.start_animation(self.SLOW_DURATION)

    def end_animation(self):
        """Finish the fade animation"""
        self.ga.stop()
        self.hide()


class QCustomQWidget (QWidget):
    """This custom widget is a helper widget to show the name of the world and an icon for the tooltip"""

    def __init__(self, world, parent=None):
        """Constructor of the class

        Arguments:
            world {str} -- Name of the world

        Keyword Arguments:
            parent {ui.gui.views.world_selection.WorldSelection} -- Parent of this widget (default: {None})
        """
        super(QCustomQWidget, self).__init__(parent)
        self.world = world
        self.parent = parent
        self.textQVBoxLayout = QVBoxLayout()
        font = QFont('Arial', 30)
        self.textUpQLabel = QLabel()
        self.textUpQLabel.setFont(font)
        self.textUpQLabel.setStyleSheet('color: white')
        self.textQVBoxLayout.addWidget(self.textUpQLabel)
        self.allQHBoxLayout = QHBoxLayout()
        self.infoIcon = InfoLabel(self.world['description'])
        self.allQHBoxLayout.addLayout(self.textQVBoxLayout)
        self.allQHBoxLayout.addWidget(self.infoIcon)
        self.setLayout(self.allQHBoxLayout)

    def setTextUp(self, text):
        """Modify the shown world name"""
        self.textUpQLabel.setText(text)

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        self.textUpQLabel.setStyleSheet('color: yellow')

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.textUpQLabel.setStyleSheet('color: white')

    def mousePressEvent(self, event):
        """Mouse event when pressing the widget"""
        if event.button() == Qt.LeftButton:
            # self.parent.generate_launch_file(self.world['world'])
            self.parent.save_config(self.parent.robot_type, self.world['world'])
            self.parent.switch_window.emit()


class WorldSelection(QWidget):
    """Main class of the world selection view. Handles all the elements of this view"""
    switch_window = pyqtSignal()

    def __init__(self, robot_type, configuration, parent=None):
        """Constructor of the class

        Arguments:
            robot_type {str} -- Identifier of the robot type (f1, drone, car, ...)
            configuration {utils.configuration.Config} -- Configuration instance of the application

        Keyword Arguments:
            parent {ui.gui.views_controller.ParentWindow} -- Parent container of this view (default: {None})
        """
        super(WorldSelection, self).__init__(parent)
        self.parent = parent
        self.robot_type = robot_type
        self.configuration = configuration
        self.enable_gazebo_gui = 'false'
        self.initUI()

    def initUI(self):
        """Initialize the GUI elements"""
        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51)')
        self.setLayout(main_layout)

        logo = Logo()

        self.v3d = View3D(self.robot_type, self)
        frame = QFrame(self)
        self.r1_layout = QHBoxLayout()

        with open(resources_path + 'worlds.json') as f:
            data = f.read()
        worlds = json.loads(data)[self.robot_type]

        myQListWidget = QListWidget()
        myQListWidget.setStyleSheet("border: 0px;")
        myQListWidget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        myQListWidget.verticalScrollBar().setSingleStep(10)
        myQListWidget.setMaximumWidth(800)

        for world in worlds:
            myQCustomQWidget = QCustomQWidget(world, self)
            myQCustomQWidget.setTextUp(world['name'])
            myQListWidgetItem = QListWidgetItem(myQListWidget)
            myQListWidgetItem.setSizeHint(QSize(100, 80))
            myQListWidget.addItem(myQListWidgetItem)
            myQListWidget.setItemWidget(myQListWidgetItem, myQCustomQWidget)

        self.r1_layout.addWidget(self.v3d)
        self.r1_layout.addWidget(myQListWidget)

        frame.setLayout(self.r1_layout)

        # enable_gui = QCheckBox(self)
        # enable_gui.setText("Gazebo GUI")
        # enable_gui.setStyleSheet("""QCheckBox { color: white; font-size: 20px;}
        #                             QCheckBox::indicator {
        #                                     border: 1px solid white;
        #                                     background: white;
        #                                     height: 10px;
        #                                     width: 10px;
        #                                     border-radius: 2px
        #                                 }
        #                           """)
        # # enable_gui.setStyleSheet("""QCheckBox; QCheckBox::indicator { width: 400px; height: 400px;}""")
        # enable_gui.stateChanged.connect(self.handle_gazebo_gui)
        # enable_gui.setFixedHeight(50)

        font = QFont('Arial', 30)
        lbl = ClickableLabel(self)
        lbl.setFont(font)
        lbl.setText("Select your world")
        lbl.setFixedHeight(100)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet('color: yellow')

        main_layout.addWidget(logo)
        main_layout.addWidget(frame)
        # main_layout.addWidget(enable_gui, 2, Qt.AlignCenter)
        main_layout.addWidget(lbl)

    # def handle_gazebo_gui(self):
    #     self.enable_gui = not self.enable_gazebo_gui

    # def generate_launch_file(self, world_name):
    #     with open(resources_path + 'template.launch') as file:
    #         data = file.read()

    #     data = data.replace('[WRLD]', world_name)
    #     data = data.replace('[GUI]', self.enable_gazebo_gui)

    #     with open('world.launch', 'w') as file:
    #         file.write(data)

    def save_config(self, robot_type, world):
        """Save the chosen configuration to the configuration instance

        Arguments:
            robot_type {str} -- Identifier of the robot type (f1, drone, car, ...)
            world {str} -- Path to the launch file of the environment
        """
        self.configuration.robot_type_set(robot_type)
        self.configuration.current_world = world
        self.configuration.create_sensors_actuators(robot_type)

    def update_gui(self):
        pass
