from PyQt5.QtCore import Qt, QPropertyAnimation, QSequentialAnimationGroup, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QLabel, QGraphicsOpacityEffect, QWidget, QGridLayout, QVBoxLayout, QFrame

from views.logo import Logo
from views.models3d import View3D


class ClickableLabel(QLabel):

    clicked = pyqtSignal()

    in_label = False
    SLOW_DURATION = 1500
    FAST_DURATION = 500

    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        self.start_animation(self.SLOW_DURATION)

    def start_animation(self, duration):
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
        self.ga.stop()
        self.start_animation(self.FAST_DURATION)

    def leaveEvent(self, event):
        self.ga.stop()
        self.start_animation(self.SLOW_DURATION)

    def end_animation(self):
        self.ga.stop()
        self.hide()


class CustomQFrame(QFrame):

    def __init__(self, scene, parent=None, flags=Qt.WindowFlags()):
        super(CustomQFrame, self).__init__(parent=parent, flags=flags)
        self.scene = scene
        self.parent = parent

    def enterEvent(self, event):
        self.setStyleSheet('QFrame {background-color: rgba(255,255,255,0.3); border: 2px solid white; }')
        self.scene.view.defaultFrameGraph().setClearColor(QColor(255, 255, 255))

    def leaveEvent(self, event):
        self.setStyleSheet('QFrame {background-color: rgb(51,51,51); border: 0px solid white; }')
        self.scene.view.defaultFrameGraph().setClearColor(QColor(51, 51, 51))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            print('mouse pressed in', self.scene.robot_type)
            self.scene.set_animation_speed(1000)
            self.scene.start_animation_with_duration(2000)
            self.parent.parent.robot_selection = self.scene.robot_type


class RobotSelection(QWidget):
    updGUI = pyqtSignal()
    switch_window = pyqtSignal()

    def __init__(self, parent=None):
        super(RobotSelection, self).__init__(parent)
        self.updGUI.connect(self.update_gui)
        self.parent = parent
        self.initUI()

    def initUI(self):
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
        label = QLabel(self)
        label.setFont(font)
        label.setText(text)
        label.setFixedHeight(40)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet('background-color: rgba(0,0,0,0);color: white; border: 0px solid black; ')
        return label

    def create_robot_frame(self, robot_type, robot_name):

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
    """ memory secure. """
    for i in reversed(range(layout.count())):
        widgetToRemove = layout.itemAt(i).widget()
        # remove it from the layout list
        layout.removeWidget(widgetToRemove)
        # remove it from the gui
        widgetToRemove.setParent(None)
