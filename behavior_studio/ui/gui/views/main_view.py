#!/usr/bin/env python
""" This module contains all the functionality of the main view of the application.

This view is the one that will allow the user to interact with the robot and their algoritmhs, offering an intuitive
and complete suite of tools for that purpose. This consists on two sides: toolbar (left) and frame matrix (right).
The toolbar offers the user a set of tools to control both the application, the robot, the simulation and the algorithm.
The frame matrix offers the user the visualization of the robot's sensors. Each frame is configurable, so the user will
be able to show different information provided by the robot's sensors in each frame. The disposal of the matrix will be
determined by either a configuration file or the actual GUI.

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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QRadioButton, QScrollArea,
                             QVBoxLayout, QWidget)

from ui.gui.views.toolbar import Toolbar
from ui.gui.views.widgets.camera import CameraWidget
from ui.gui.views.widgets.laser import LaserWidgetPro

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'


class InfoLabel(QLabel):
    """Class that extends the functionality of the default QLabel adding a tooltip"""

    def __init__(self, description='Click me!', parent=None):
        """Constructor of the class

        Arguments:
            description {str} -- Tooltip text
            parent {ui.gui.views.main_view.MainView} -- Parent of the widget

        """
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
        """Mouse event when entering the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(60, 60, Qt.KeepAspectRatio))

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        """Mouse event when pressing the widget"""
        if event.button() & Qt.LeftButton:
            self.parent.show_information_popup()


class HLine(QFrame):
    """Helper class that implements an horizontal separator"""
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine | self.Sunken)
        pal = self.palette()
        pal.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setPalette(pal)


class ClickableLabel(QLabel):
    """Class that extends the functionality of the default QLabel adding the mouse event handlers

    This class is used to get back to the configuration view in each frame by adding a gear icon in the top left corner
    of the frame, so one can change the configuration of the frame whenever needed.
    """

    def __init__(self, creator, parent=None):
        """Constructor of the class

        Arguments:
            creator {ui.gui.views.main_view.ClickableQFrame} -- Creator of this widget

        Keyword Arguments:
            parent {ui.gui.views.widget.*} -- Widget that shows the data (CameraWidget, etc) (default: {None})
        """
        QLabel.__init__(self, parent)
        self.setFixedSize(30, 30)
        self.parent = parent
        self.creator = creator
        self.pmax_dark = QPixmap(':/assets/gear_dark.png')
        self.pmax_light = QPixmap(':/assets/gear_light.png')
        self.setStyleSheet('background-color: rgba(0, 0, 0, 0)')
        self.setPixmap(self.pmax_light)
        self.setScaledContents(True)

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        self.setPixmap(self.pmax_dark)

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        self.setPixmap(self.pmax_light)

    def mousePressEvent(self, event):
        """Mouse event when pressing the widget"""
        if event.button() & Qt.LeftButton:
            self.creator.clear()


class FrameConfig(QWidget):
    """This class contains the tools for configuring each sensor frame

    Attributes:
        frame_name {str} -- Identifier of the frame
        data_type {str} -- Identifier of the type of data to be represented
    """

    def __init__(self, parent=None):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views.main_view.ClickableQFrame} -- Parent of this widget (default: {None})
        """
        QWidget.__init__(self, parent)
        self.parent = parent
        self.frame_name = ''
        self.data_type = None
        self.initUI()

    def initUI(self):
        """Initializes the widgets elements (buttons, and tools in general)"""

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
        """Callback that handles the radio boxes of the data type

        Whenever a radiobox is selected, the data type of this widget will change

        Arguments:
            button {} -- Radio button that triggered the callback
        """
        self.parent.data_type = button.objectName()
        self.change_title()

    def confirm_configuration(self):
        """Callback for confirmation button

        When all the settings are setted up, this button will hide the configuration view
        """
        self.hide()
        self.parent.keep_ratio = self.aspect_ratio.isChecked
        self.parent.confirm.emit()

    def change_name(self, text):
        """Callback that handles changes in the textbox

        When the user updates the textbox with new text, this callback will be invoked
        """
        prev_id = self.parent.id
        self.parent.id = "_".join(text.split())
        self.parent.setObjectName(self.parent.id)
        self.parent.parent.change_frame_name(prev_id, self.parent.id)
        self.change_title()

    def change_title(self):
        """This function will change the frame id"""
        self.parent.setTitle('id:  ' + self.parent.id + ' | Data:  ' + str(self.parent.data_type))


class ClickableQFrame(QGroupBox):
    """This class contains the logic for visualizing each sensor frame

    This class will create the visualization of the sensor based on the data type provided by
    QFrameConfig class. Depending on this field the frame will represent the information with different
    widgets (see ui.gui.views.widgets)

    """

    # default style
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
    # hover style
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

    def __init__(self, frame_id, data='rgbimage', parent=None):
        """Constructor of the class

        Arguments:
            frame_id {str} -- Identificator of the frame

        Keyword Arguments:
            data {str} -- Identificator of the data type (could be 'depthimage, laser and pose) (default: {'rgbimage'})
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget (default: {None})
        """

        QGroupBox.__init__(self)
        self.parent = parent
        self.id = frame_id
        self.data_type = data
        self.setObjectName(frame_id)
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
        """Function that creates the widget based on the data_type.

        If the data_type is 'rgbimage' this will create a CameraWidget (ui.gui.views.widgets.camera.CameraWidget)
        if the data_type is 'laser' this will create a LaserWidget (ui.gui.views.widgets.laser.LaserWidget)
        and so on.
        """
        self.scroll.hide()
        if self.data_type == 'rgbimage':
            self.widget = CameraWidget(self.objectName(), self.width(), self.height(), self.keep_ratio, self.parent)
        elif self.data_type == 'depthimage':
            pass
        elif self.data_type == 'laser':
            self.widget = LaserWidgetPro(self.objectName(), self.width(), self.height(), self.parent)
        elif self.data_type == 'pose':
            # TODO: implement pose3D widget
            pass
        self.lay.addWidget(self.widget)
        self.settings_label = ClickableLabel(self, self.widget)
        self.settings_label.move(10, 10)

    def clear(self):
        """Function that clear all the configuration of the frame and shows the configuration view to reconfigure it."""

        self.widget.close()
        self.widget.setParent(None)
        self.scroll.show()
        self.frame_config.show()

    def update(self):
        """Update function that will refresh the GUI with the GUI loop"""

        if self.widget:
            self.widget.update()
        pass


class LayoutMatrix(QWidget):
    """Class that contains the logic to create the different sensor frames in a matrix fashion

    Given a layout configuration, this class will create and setup all the frames in that layout as defined in the
    configuration.
    """

    def __init__(self, positions, configuration, parent=None):
        """Constructor of the class

        Arguments:
            positions {list} -- List of positions of each frame. Each element on the list is a frame configuration.
            configuration {utils.configuration.Config} -- Configuration instance of the application

        Keyword Arguments:
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget (default: {None})
        """
        QWidget.__init__(self, parent)
        self.main_layout = QGridLayout()
        self.parent = parent

        self.setStyleSheet('background-color: rgb(51,51,51)')
        self.setLayout(self.main_layout)

        # coming from title window
        if positions:
            for idx, c in enumerate(positions):
                sensor_frame = ClickableQFrame('frame_' + str(c[4]), data='rgbimage', parent=self.parent)
                self.main_layout.addWidget(sensor_frame, c[0], c[1], c[2], c[3])
        # coming from config file
        else:
            layout = configuration.layout
            for frame in layout:
                sensor_frame = ClickableQFrame(frame, data=layout[frame][1], parent=self.parent)
                c = layout[frame][0]
                self.main_layout.addWidget(sensor_frame, c[0], c[1], c[2], c[3])

    def update(self):
        """Update the matrix of frames with the GUI loop."""
        for i in range(self.main_layout.count()):
            self.main_layout.itemAt(i).widget().update()


class MainView(QWidget):
    """Class that handles all the elements of the main view of the application

    Attributes:
        parent {ui.gui.views_controller.ParentWindow} -- Parent of this widget
        controller {utils.controller.Controller} -- Controller of the application
        layout_configuration {list} -- List of positions of the frame matrix
        configuration {utils.configuration.Config} -- Configuration instance of the application
    """
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
        """Initialize GUi's elements"""

        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51); color: white')
        self.setLayout(main_layout)

        # define view's widgets

        central_layout = QHBoxLayout()
        toolbar = Toolbar(self.configuration, self.controller, self.parent)
        self.matrix = LayoutMatrix(self.layout_configuration, self.configuration, self)
        central_layout.addWidget(toolbar)
        central_layout.addWidget(self.matrix)

        # main_layout.addWidget(logo)
        main_layout.addLayout(central_layout)

        self.show_information_popup()

    def get_frame(self, frame_id):
        """Get an instance of the frame by it's identificator

        Arguments:
            frame_id {str} -- Identificator of the frame

        Returns:
            ui.gui.views.main_view.ClickableQFrame -- Instance of the found frame
        """
        return self.matrix.findChild(ClickableQFrame, frame_id)

    def update_gui(self):
        """Update GUI with the GUI's refresh loop"""
        self.matrix.update()

    def change_frame_name(self, old, new):
        """Change the identificator of one frame

        Arguments:
            old {str} -- Old frame identificator
            new {str} -- New frame identificator
        """
        self.configuration.change_frame_name(old, new)

    def show_information_popup(self):
        pass
