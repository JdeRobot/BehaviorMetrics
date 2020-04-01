from PyQt5.QtCore import (QPropertyAnimation, QSequentialAnimationGroup, Qt, pyqtSignal)
from PyQt5.QtGui import QColor, QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QGraphicsOpacityEffect, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton, QScrollArea, QVBoxLayout,
                             QWidget)

from logo import Logo
from widgets.camera import CameraWidget
from widgets.laser import LaserWidgetPro
from toolbar import Toolbar


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


class HLine(QFrame):
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine | self.Sunken)
        pal = self.palette()
        pal.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setPalette(pal)


class ClickableLabel(QLabel):

    def __init__(self, creator, parent=None):
        QLabel.__init__(self, parent)
        self.setFixedSize(30, 30)
        self.parent = parent
        self.creator = creator
        self.pmax_dark = QPixmap(':/assets/gear_icon_dark.png')
        self.pmax_light = QPixmap(':/assets/gear_icon_light.png')
        self.setStyleSheet('background-color: rgba(0, 0, 0, 0)')
        self.setPixmap(self.pmax_light)
        self.setScaledContents(True)

    def enterEvent(self, event):
        self.setPixmap(self.pmax_dark)

    def leaveEvent(self, event):
        self.setPixmap(self.pmax_light)

    def mousePressEvent(self, event):
        if event.button() & Qt.LeftButton:
            self.creator.clear()


class FrameConfig(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.parent = parent
        self.frame_name = ''
        self.data_type = None
        self.msg = ''
        self.initUI()

    def initUI(self):

        self.setFixedSize(250, 300)
        self.setStyleSheet('color: white; font-size: 14px')
        layout = QVBoxLayout()
        name_label = QLabel('Frame name')
        name_edit = QLineEdit()
        name_edit.setPlaceholderText('Frame name')
        name_edit.setMaximumWidth(200)
        data_label = QLabel('Data type')
        self.button_group = QButtonGroup()
        self.button_group.setExclusive(True)
        rgb_image_data = QRadioButton('RGB Image')
        rgb_image_data.setObjectName('rgbimage')
        depth_image_data = QRadioButton('Depth Image')
        depth_image_data.setObjectName('depthimage')
        laser_data = QRadioButton('Laser Data')
        laser_data.setObjectName('laser')
        pose_data = QRadioButton('Pose3D Data')
        pose_data.setObjectName('pose')
        rgb_image_data.setChecked(True)
        depth_image_data.setEnabled(False)
        self.button_group.addButton(rgb_image_data)
        self.button_group.addButton(laser_data)
        self.button_group.addButton(pose_data)
        self.button_group.addButton(depth_image_data)
        self.aspect_ratio = QCheckBox('Keep aspect ratio')
        self.aspect_ratio.setChecked(True)
        confirm_button = QPushButton('Confirm')
        confirm_button.setMaximumWidth(200)

        self.button_group.buttonClicked.connect(self.button_handle)
        confirm_button.pressed.connect(self.confirm_configuration)
        name_edit.textChanged[str].connect(self.change_name)

        layout.addWidget(name_label, alignment=Qt.AlignCenter)
        layout.addWidget(name_edit)
        layout.addWidget(data_label, alignment=Qt.AlignCenter)
        layout.addWidget(rgb_image_data)
        layout.addWidget(depth_image_data)
        layout.addWidget(laser_data)
        layout.addWidget(pose_data)
        layout.addWidget(HLine())
        layout.addWidget(self.aspect_ratio, alignment=Qt.AlignLeft)
        layout.addWidget(confirm_button)

        self.setLayout(layout)

    def button_handle(self, button):
        self.parent.data_type = button.objectName()
        self.change_title()

    def confirm_configuration(self):
        self.hide()
        self.parent.keep_ratio = self.aspect_ratio.isChecked
        self.parent.confirm.emit()

    def change_name(self, text):
        prev_id = self.parent.id
        self.parent.id = "_".join(text.split())
        self.parent.setObjectName(self.parent.id)
        self.parent.parent.change_frame_name(prev_id, self.parent.id)
        self.change_title()

    def change_title(self):
        self.parent.setTitle('id:  ' + self.parent.id + ' | Data:  ' + str(self.parent.data_type))


class ClickableQFrame(QGroupBox):

    normal_qss = """
                    QGroupBox {
                        font: bold;
                        border: 1px solid silver;
                        border-radius: 6px;
                        margin-top: 6px;
                    }

                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 7px;
                        padding: 0px 5px 0px 5px;
                    }
                """
    hover_qss = """
                    QGroupBox{
                        border: 3px solid yellow;
                        border-radius: 5px;
                        font-size: 18px;
                        font-weight: bold;
                    }
                    QGroupBox::Title{
                        color: white;
                        background-color: rgba(51, 51, 51);
                    }
                    QLabel {
                        font-size: 50px;
                        color: white
                    }
                """

    confirm = pyqtSignal()

    def __init__(self, id, parent=None):
        QGroupBox.__init__(self)
        self.parent = parent
        self.id = id
        self.data_type = 'rgbimage'
        self.setObjectName(id)
        self.setTitle('id:  ' + self.id + ' | Data:  ' + str(self.data_type))
        self.setAlignment(Qt.AlignCenter)
        self.is_active = False
        self.setStyleSheet(self.normal_qss)
        self.confirm.connect(self.create_widget)
        self.widget = None
        self.keep_ratio = False

        self.lay = QHBoxLayout()
        self.frame_config = FrameConfig(self)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.frame_config)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.lay.addWidget(self.scroll, alignment=Qt.AlignCenter)

        self.setLayout(self.lay)

    def create_widget(self):
        self.scroll.hide()
        if self.data_type == 'rgbimage':
            self.widget = CameraWidget(self.objectName(), self.width(), self.height(), self.keep_ratio, self.parent)
        elif self.data_type == 'depthimage':
            pass
        elif self.data_type == 'laser':
            self.widget = LaserWidgetPro(self.objectName(), self.width(), self.height(), self.parent)
        elif self.data_type == 'pose':
            pass
        self.lay.addWidget(self.widget)
        self.settings_label = ClickableLabel(self, self.widget)
        self.settings_label.move(10, 10)

    def clear(self):
        self.widget.close()
        self.widget.setParent(None)
        self.scroll.show()
        self.frame_config.show()

    def update(self):
        if self.widget:
            self.widget.update()
        pass


class FakeToolbar(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setMaximumWidth(400)
        self.initUI()

        self.setStyleSheet( """
                    QGroupBox {
                        font: bold;
                        border: 1px solid silver;
                        border-radius: 6px;
                        margin-top: 6px;
                    }

                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 7px;
                        padding: 0px 5px 0px 5px;
                    }
                """)

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

        for idx, c in enumerate(positions):
            sensor_frame = ClickableQFrame('frame_' + str(c[4]), self.parent)
            self.main_layout.addWidget(sensor_frame, c[0], c[1], c[2], c[3])

    def update(self):
        for i in range(self.main_layout.count()):
            self.main_layout.itemAt(i).widget().update()


class MainView(QWidget):

    switch_window = pyqtSignal()

    def __init__(self, layout_configuration, configuration, controller, parent=None):
        super(MainView, self).__init__(parent)
        self.parent = parent
        self.controller = controller
        self.layout_configuration = layout_configuration
        self.configuration = configuration
        self.parent.status_bar.showMessage('Select the topic for each view')
        self.initUI()

    def initUI(self):

        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51); color: white')
        self.setLayout(main_layout)

        # define view's widgets

        central_layout = QHBoxLayout()
        toolbar = Toolbar(self.configuration, self.controller)
        self.matrix = LayoutMatrix(self.layout_configuration, self)
        central_layout.addWidget(toolbar)
        central_layout.addWidget(self.matrix)

        # main_layout.addWidget(logo)
        main_layout.addLayout(central_layout)

        self.show_information_popup()

    def get_frame(self, id):
        return self.matrix.findChild(ClickableQFrame, id)

    def update_gui(self):
        self.matrix.update()

    def change_frame_name(self, old, new):
        self.configuration.change_frame_name(old, new)

    def show_information_popup(self):
        pass
