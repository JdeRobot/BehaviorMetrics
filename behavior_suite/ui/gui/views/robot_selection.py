#!/usr/bin/env python
""" This module is responsible for handling the logic of the robot and its current brain.

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

from PyQt5.QtCore import (QPropertyAnimation, QSequentialAnimationGroup, Qt,
                          pyqtSignal)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QFrame, QGraphicsOpacityEffect, QGridLayout,
                             QLabel, QVBoxLayout, QWidget)

from ui.gui.views.logo import Logo
from ui.gui.views.models3d import View3D

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class ClickableLabel(QLabel):
    """Class that extends the functionality of the default QLabel adding mouse events."""

    clicked = pyqtSignal()

    in_label = False
    SLOW_DURATION = 1500
    FAST_DURATION = 500

    def __init__(self, parent=None):
        """Constructor of the class"""
        QLabel.__init__(self, parent)
        self.start_animation(self.SLOW_DURATION)

    def start_animation(self, duration):
        """Start fade animation for the label

        Arguments:
            duration {int} -- Duration in milliseconds of a complete fadein-fadeout cycle
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
        """End the fade animation"""
        self.ga.stop()
        self.hide()


class CustomQFrame(QFrame):
    """This class is a container of the 3D previsualization models of the robots"""

    def __init__(self, scene, parent=None, flags=Qt.WindowFlags()):
        """Constructor of the class

        Arguments:
            scene {ui.gui.views.models3d.View3D} -- 3DView instance.

        Keyword Arguments:
            parent {ui.gui.views.robot_selection.RobotSelection} -- Parent of this widget (default: {None})
        """
        super(CustomQFrame, self).__init__(parent=parent, flags=flags)
        self.scene = scene
        self.parent = parent

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        self.setStyleSheet('QFrame {background-color: rgba(255,255,255,0.3); border: 2px solid white; }')
        self.scene.view.defaultFrameGraph().setClearColor(QColor(255, 255, 255))

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.setStyleSheet('QFrame {background-color: rgb(51,51,51); border: 0px solid white; }')
        self.scene.view.defaultFrameGraph().setClearColor(QColor(51, 51, 51))

    def mousePressEvent(self, event):
        """Mouse event when pressing the widget"""
        if event.button() == Qt.LeftButton:
            print('mouse pressed in', self.scene.robot_type)
            self.scene.set_animation_speed(1000)
            self.scene.start_animation_with_duration(2000)
            self.parent.parent.robot_selection = self.scene.robot_type


class RobotSelection(QWidget):
    """Main class of the robot selector view"""
    switch_window = pyqtSignal()

    def __init__(self, parent=None):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views_controller.ParentWindow} -- Parent container of this widget (default: {None})
        """
        super(RobotSelection, self).__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        """Initialize GUI elements"""
        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51)')
        self.setLayout(main_layout)

        logo = Logo()
        main_layout.addWidget(logo)

        self.robot_layout = QGridLayout()
        self.f1_frame = self.create_robot_frame('f1', 'Formula 1')
        self.drone_frame = self.create_robot_frame('drone', 'Drone')
        self.roomba_frame = self.create_robot_frame('roomba', 'Roomba')
        self.car_frame = self.create_robot_frame('car', 'Car')
        self.turtle_frame = self.create_robot_frame('turtlebot', 'Turtlebot')
        self.pepper_frame = self.create_robot_frame('pepper', 'Pepper - Comming soon')

        self.robot_layout.addWidget(self.f1_frame, 0, 0)
        self.robot_layout.addWidget(self.drone_frame, 0, 1)
        self.robot_layout.addWidget(self.roomba_frame, 0, 2)
        self.robot_layout.addWidget(self.car_frame, 1, 0)
        self.robot_layout.addWidget(self.turtle_frame, 1, 1)
        self.robot_layout.addWidget(self.pepper_frame, 1, 2)

        font = QFont('Arial', 30)
        lbl = ClickableLabel(self)
        lbl.setFont(font)
        lbl.setText("Select your robot")
        lbl.setFixedHeight(100)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet('color: yellow')

        main_layout.addLayout(self.robot_layout)
        main_layout.addWidget(lbl)

    def createLabel(self, text, font):
        """Generic function to create a label with a custom text and font

        Arguments:
            text {str} -- Text to be shown (robot type)
            font {QFont} -- Font for the text

        Returns:
            QLabel -- Instance of set up label
        """
        label = QLabel(self)
        label.setFont(font)
        label.setText(text)
        label.setFixedHeight(40)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('background-color: rgba(0,0,0,0);color: white; border: 0px solid black; ')
        return label

    def create_robot_frame(self, robot_type, robot_name):
        """Generic function to create a robot frame.

        Includes the frame of the 3D visualization and the robot type label.

        Arguments:
            robot_type {str} -- Identifier of the robot type (f1, drone, car, ...)
            robot_name {str} -- Name of the robot (typically same as robot_type)

        Returns:
            ui.gui.views.robot_selection.CustomFrame -- A complete set up robot 3D visualization frame
        """

        v3d = View3D(robot_type, self)
        frame = CustomQFrame(v3d, self)
        r1_layout = QVBoxLayout()
        font = QFont('Arial', 15)
        lr = self.createLabel(robot_name, font)

        r1_layout.addWidget(v3d)
        r1_layout.addWidget(lr)
        frame.setLayout(r1_layout)

        return frame

    def emit_and_destroy(self):
        """Safely destroying this view.
        TODO: sometimes it fails with a core dumped"""
        self.f1_frame.scene.stop_animation()
        self.drone_frame.scene.stop_animation()
        self.roomba_frame.scene.stop_animation()
        self.car_frame.scene.stop_animation()
        self.turtle_frame.scene.stop_animation()
        self.pepper_frame.scene.stop_animation()

        self.switch_window.emit()
        self.f1_frame.scene.view.deleteLater()
        self.drone_frame.scene.view.deleteLater()
        self.roomba_frame.scene.view.deleteLater()
        self.car_frame.scene.view.deleteLater()
        self.turtle_frame.scene.view.deleteLater()
        self.pepper_frame.scene.view.deleteLater()

    def update_gui(self):
        pass


def delete_widgets_from(layout):
    """ Memory secure deletion of widgets. """
    for i in reversed(range(layout.count())):
        widgetToRemove = layout.itemAt(i).widget()
        # remove it from the layout list
        layout.removeWidget(widgetToRemove)
        # remove it from the gui
        widgetToRemove.setParent(None)
