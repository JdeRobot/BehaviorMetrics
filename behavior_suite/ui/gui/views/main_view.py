from PyQt5.QtCore import Qt, QPropertyAnimation, QSequentialAnimationGroup, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import (QLabel, QFrame, QGraphicsOpacityEffect, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QApplication, QWidget, QPushButton, QGroupBox)

from logo import Logo

current_selection_id = 0


class InfoLabel(QLabel):

    def __init__(self, description='Click me!', parent=None):
        QLabel.__init__(self, parent)
        self.description = description
        self.parent = parent

        self.setFixedSize(60, 60)
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))
        self.setMouseTracking(True)
        self.setStyleSheet("""QToolTip {
                           background-color: rgb(51,51,51);
                           color: white;
                           border: black solid 1px;
                           font-size: 20px;
                           }""")
        self.setToolTip(self.description)

    def enterEvent(self, event):
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(60, 60, Qt.KeepAspectRatio))

    def leaveEvent(self, event):
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        if event.button() & Qt.LeftButton:
            self.parent.show_information_popup()


class AnimatedLabel(QLabel):

    SLOW_DURATION = 1500
    FAST_DURATION = 500

    def __init__(self, parent=None, color='yellow'):
        QLabel.__init__(self, parent)
        self.start_animation(self.SLOW_DURATION)
        font = QFont('Arial', 30)
        self.setFont(font)
        self.setText("Select your layout")
        self.setFixedHeight(100)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: ' + color)

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


class ClickableQFrame(QFrame):

    normal_qss = """
                    QFrame#customframe{
                        border: 3px solid white;
                        border-radius: 20px
                    }
                    QLabel {
                        font-size: 50px;
                        color: white
                    }
                """

    hover_qss = """
                    QFrame#customframe{
                        border: 3px solid yellow;
                        border-radius: 20px
                    }
                    QLabel {
                        font-size: 50px;
                        color: white
                    }
                """

    def __init__(self, parent=None):
        QFrame.__init__(self)
        self.setObjectName('customframe')
        self.parent = parent
        self.is_active = False
        self.topic = None
        self.setStyleSheet(self.normal_qss)
        lay = QVBoxLayout()
        global current_selection_id
        self.labelgr = QLabel('Click me!')
        self.labelgr.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.labelgr)
        self.setLayout(lay)

    def enterEvent(self, event):
        if not self.is_active:
            self.setStyleSheet(self.hover_qss)

    def leaveEvent(self, event):
        if not self.is_active:
            self.setStyleSheet(self.normal_qss)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            # show frame configuration window
            pass

    def clear(self):
        self.is_active = False
        self.topic = None
        self.labelgr.setText('Click me!')
        self.setStyleSheet(self.normal_qss)


class FakeToolbar(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setMaximumWidth(400)
        self.initUI()

        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid gray;
                border-color: #ffffff00;
                margin-top: 27px;
                font-size: 14px;
                border-radius: 15px;
            }
            QGroupBox::title {
                border-top-left-radius: 9px;
                border-top-right-radius: 9px;
                padding: 2px 82px;
                color: #ffffff00;
            }""")

    def initUI(self):
        self.main_layout = QVBoxLayout()
        b1 = QGroupBox()
        b1.setTitle('Toolbar')
        logo = Logo()
        self.main_layout.addWidget(b1)
        self.main_layout.addWidget(logo)
        self.setLayout(self.main_layout)


class LayoutMatrix(QWidget):

    def __init__(self, positions, parent=None):
        QWidget.__init__(self, parent)
        self.main_layout = QGridLayout()
        self.parent = parent

        self.setStyleSheet('background-color: rgb(51,51,51)')
        self.setLayout(self.main_layout)

        for c in positions:
            sensor_frame = ClickableQFrame(self.parent)
            self.main_layout.addWidget(sensor_frame, c[0], c[1], c[2], c[3])


class MainView(QWidget):
    updGUI = pyqtSignal()
    switch_window = pyqtSignal()

    def __init__(self, layout_configuration, parent=None):
        super(MainView, self).__init__(parent)
        self.updGUI.connect(self.update_gui)
        self.parent = parent
        self.layout_configuration = layout_configuration
        self.parent.status_bar.showMessage('Select the topic for each view')
        self.initUI()

    def initUI(self):

        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51); color: white')
        self.setLayout(main_layout)

        # define view's widgets

        central_layout = QHBoxLayout()
        toolbar = FakeToolbar()
        matrix = LayoutMatrix(self.layout_configuration, self)
        central_layout.addWidget(toolbar)
        central_layout.addWidget(matrix)

        # main_layout.addWidget(logo)
        main_layout.addLayout(central_layout)

        self.show_information_popup()

    def update_gui(self):
        pass

    def show_information_popup(self):
        pass
