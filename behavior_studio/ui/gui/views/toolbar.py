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

import json
import os

import rospy
from PyQt5.QtCore import (QPropertyAnimation, QSequentialAnimationGroup, QSize,
                          Qt)
from PyQt5.QtGui import QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                             QFrame, QGraphicsOpacityEffect, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QScrollArea, QSpacerItem,
                             QVBoxLayout, QWidget)

from ui.gui.views.logo import Logo
from ui.gui.views.social import SocialMedia
from utils import constants, environment

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

worlds_path = constants.ROOT_PATH + '/ui/gui/resources/worlds.json'
brains_path = constants.ROOT_PATH + '/brains/'

""" TODO:   put button for showing logs? """


class TopicsPopup(QWidget):
    """This class will show a popup window to select topics to be recorded in a rosbag

    Attributes:
        active_topics {list} -- List of topcis to be recorded"""

    def __init__(self):
        """Construtctor of the class"""
        QWidget.__init__(self)
        self.setFixedSize(800, 600)
        self.setWindowTitle("Select your topics")
        self.active_topics = []
        self.hide()
        self.initUI()

    def initUI(self):
        """Initialize GUI elements"""
        self.setStyleSheet('background-color: rgb(51, 51, 51); color: white;')

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.button_group = QButtonGroup()
        self.button_group.buttonClicked.connect(self.add_topic)
        self.button_group.setExclusive(False)

        scroll = QScrollArea(self)
        self.main_layout.addWidget(scroll)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget(scroll)

        self.scroll_layout = QVBoxLayout(scroll_content)
        scroll_content.setLayout(self.scroll_layout)
        scroll.setWidget(scroll_content)

    def fill_topics(self):
        """Fill the active_topics with all the topics selected by the user"""
        topics = rospy.get_published_topics()
        for idx, topic in enumerate(topics):
            cont = QFrame()
            ll = QHBoxLayout()
            cont.setLayout(ll)
            topic_check = QCheckBox(str(topic[0]))
            self.button_group.addButton(topic_check)
            ll.addWidget(topic_check)
            self.scroll_layout.addWidget(cont)
            if idx % 2 == 0:
                cont.setStyleSheet('background-color: rgb(51,51,51)')
            else:
                cont.setStyleSheet('background-color: rgb(71, 71, 71)')

    def show_updated(self):
        """Update the window"""
        self.show()
        self.fill_topics()
        self.active_topics = []

    def add_topic(self, btn):
        """Callback that adds topic to the active_topics list"""
        self.active_topics.append(btn.text())


class HLine(QFrame):
    """Helper class that creates an horizontal separator"""
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine | self.Sunken)
        pal = self.palette()
        pal.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setPalette(pal)


class AnimatedLabel(QLabel):
    """Class that extends the default functionality of the QLabel adding an animation to it"""

    SLOW_DURATION = 1500
    MID_DURATION = 1000
    FAST_DURATION = 500

    def __init__(self, parent=None, color='yellow'):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views.toolbar.ToolBar} -- Parent of this widget (default: {None})
            color {str} -- Color name for the label (default: {'yellow'})
        """
        QLabel.__init__(self, parent)
        self.config_animation(self.MID_DURATION)
        self.setPixmap(QPixmap(':/assets/recording.png'))
        self.setFixedSize(40, 40)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: ' + color)
        self.setScaledContents(True)

    def config_animation(self, duration):
        """Start a fading animation for this label

        Arguments:
            duration {int} -- Duration in milliseconds of a complete fadein-fadeout cycle
        """
        self.effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.effect)

        self.fadeout = QPropertyAnimation(self.effect, b"opacity")
        self.fadeout.setDuration(duration)
        self.fadeout.setStartValue(1)
        self.fadeout.setEndValue(0)

        self.fadein = QPropertyAnimation(self.effect, b"opacity")
        self.fadein.setDuration(duration)
        self.fadein.setStartValue(0)
        self.fadein.setEndValue(1)

        self.group_animation = QSequentialAnimationGroup()
        self.group_animation.addAnimation(self.fadein)
        self.group_animation.addAnimation(self.fadeout)
        self.group_animation.setLoopCount(-1)

    def start_animation(self):
        self.group_animation.start()

    def stop_animation(self):
        self.group_animation.stop()


class ClickableLabel(QLabel):
    """Class that extends the default functionality of QLabel adding mouse events to it

    This class will handle all the buttons of the Toolbar such as star and stop buttons. It's a conversion between
    labels an buttons"""

    def __init__(self, id, size, pmap, parent=None):
        """Constructor of the class

        Parameters:
            id {str} -- Id of the label
            size {QSize} -- Size of the icon contained in the label
            pmap {QPixmap} -- Image of the icon that will be contained in the label
            parent {ui.gui.views.toolbar.ToolBar} -- Parent of this widget (default: {None})
        """

        QLabel.__init__(self, parent)
        self.setMaximumSize(size, size)
        self.parent = parent
        self.setStyleSheet("""
                            QToolTip {
                                background-color: rgb(51,51,51);
                                color: white;
                                border: black solid 1px;
                                font-size: 20px;
                            }""")
        self.setPixmap(pmap)
        self.setScaledContents(True)
        self.id = id
        self.active = False

    def enterEvent(self, event):
        """Mouse event when entering the widget"""
        # self.setStyleSheet('background-color: black')
        if self.id == 'gzcli':
            self.setPixmap(QPixmap(':/assets/gazebo_dark.png'))
        elif self.id == 'play' or self.id == 'sim':
            if self.active:
                self.setPixmap(QPixmap(':/assets/pause_dark.png'))
            else:
                self.setPixmap(QPixmap(':/assets/play_dark.png'))
        elif self.id == 'reset':
            self.setPixmap(QPixmap(':/assets/reload_dark.png'))
        pass

    def leaveEvent(self, event):
        """Mouse event when leaving the widget"""
        # self.setStyleSheet('background-color: rgb(0, 0, 0, 0,)')
        if self.id == 'gzcli':
            self.setPixmap(QPixmap(':/assets/gazebo_light.png'))
        elif self.id == 'play' or self.id == 'sim':
            if self.active:
                self.setPixmap(QPixmap(':/assets/pause.png'))
            else:
                self.setPixmap(QPixmap(':/assets/play.png'))
        elif self.id == 'reset':
            self.setPixmap(QPixmap(':/assets/reload.png'))
        pass

    def mousePressEvent(self, event):
        """Mouse event when pressing the widget"""
        if event.button() & Qt.LeftButton:
            if self.id == 'play':
                if not self.active:
                    self.setPixmap(QPixmap(':/assets/pause.png'))
                    self.active = True
                    self.parent.start_recording()
                else:
                    self.setPixmap(QPixmap(':/assets/play.png'))
                    self.active = False
                    self.parent.stop_recording()
            elif self.id == 'sim':
                if not self.active:
                    self.setPixmap(QPixmap(':/assets/pause.png'))
                    self.active = True
                    self.parent.resume_simulation()
                else:
                    self.setPixmap(QPixmap(':/assets/play.png'))
                    self.active = False
                    self.parent.pause_simulation()
            elif self.id == 'reset':
                self.parent.reset_simulation()
            elif self.id == 'gzcli':
                self.parent.open_close_simulator_gui()


class Toolbar(QWidget):
    """Main class for the toolbar widget"""

    def __init__(self, configuration, controller, parent=None):
        """Constructor of the class

        Parameters:
            configuration {utils.configuration.Config} -- Configuration instance of the application
            controller {uitls.controller.Controller} -- Controller instance of the application
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget
        """
        super(Toolbar, self).__init__()
        # self.setStyleSheet('background-color: rgb(51,51,51); color: white;')
        self.windowsize = QSize(440, 1000)
        self.configuration = configuration
        self.controller = controller
        self.parent = parent
        self.setFixedSize(self.windowsize)
        self.initUI()

        # Style of the widget
        self.setStyleSheet("""
                    QWidget {
                        background-color: rgb(51,51,51);
                        color: white;
                    }
                    QLabel {
                        color: white;
                        font-weight: normal;
                    }
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
        """Initialize GUI elements"""
        self.main_layout = QVBoxLayout()

        self.create_stats_gb()
        self.main_layout.addWidget(HLine())
        self.create_dataset_gb()
        self.main_layout.addWidget(HLine())
        self.create_brain_gb()
        self.main_layout.addWidget(HLine())
        self.create_simulation_gb()
        self.main_layout.addWidget(HLine())

        self.topics_popup = TopicsPopup()

        logo = Logo()
        social = SocialMedia()

        self.main_layout.addWidget(social)
        self.main_layout.addWidget(logo)
        self.setLayout(self.main_layout)

    def create_stats_gb(self):
        """Create stats groupbox.

        The stat groupbox will show information of the application
        TODO: complete this groupbox"""
        stats_group = QGroupBox()
        # stats_group.setMinimumHeight(400)
        stats_group.setTitle('Stats')
        stats_layout = QGridLayout()

        stats_group.setLayout(stats_layout)
        self.main_layout.addWidget(stats_group)

    def create_dataset_gb(self):
        """Creates the dataset controls groupbox."""
        dataset_group = QGroupBox()
        dataset_group.setTitle('Dataset')
        dataset_layout = QGridLayout()
        save_path_label = QLabel('Save path:  ')
        self.file_selector_save = QLineEdit()
        self.file_selector_save.setPlaceholderText('Select dataset save path')
        self.file_selector_save.setObjectName("dataset_save")
        self.file_selector_save.setReadOnly(True)

        if self.configuration.dataset_in:
            if not os.path.isfile(self.configuration.dataset_in):
                open(self.configuration.dataset_in, 'w').close()
            self.file_selector_save.setText(self.configuration.dataset_in)
        selector_save_button = QPushButton('...')
        selector_save_button.setMaximumSize(30, 30)
        selector_save_button.clicked.connect(self.saveFileDialog)
        # selector_save_button.clicked.connect(lambda: self.selectFile(self.file_selector_save))
        selector_topics_button = QPushButton('Select Topics')
        selector_topics_button.clicked.connect(lambda: self.topics_popup.show_updated())
        self.dataset_hint_label = QLabel('Select a .bag file to save dataset first!')
        self.dataset_hint_label.setStyleSheet('color: yellow; font-size: 12px; font-style: italic')
        self.dataset_hint_label.hide()

        # verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        icons_layout = QHBoxLayout()

        self.recording_animation_label = AnimatedLabel()
        self.recording_label = QLabel("Recording...")
        self.recording_label.setStyleSheet('color: yellow; font-weight: bold;')
        self.recording_animation_label.hide()
        self.recording_label.hide()
        self.recording_animation_label.setPixmap(QPixmap(':/assets/recording.png'))
        self.start_pause_record_label = ClickableLabel('play', 50, QPixmap(':/assets/play.png'), parent=self)
        self.start_pause_record_label.setToolTip('Start/Stop recording dataset')

        icons_layout.addWidget(self.recording_animation_label, alignment=Qt.AlignBottom)
        icons_layout.addWidget(self.recording_label, alignment=Qt.AlignBottom)
        # icons_layout.addItem(horizontalSpacer)
        icons_layout.addWidget(self.start_pause_record_label, alignment=Qt.AlignBottom)

        dataset_layout.addWidget(save_path_label, 0, 0, 1, 1)
        dataset_layout.addWidget(self.file_selector_save, 0, 1, 1, 1)
        dataset_layout.addWidget(selector_save_button, 0, 2, 1, 1)

        dataset_layout.addWidget(selector_topics_button, 1, 1, 1, 1)

        # dataset_layout.addItem(verticalSpacer,2,0)
        dataset_layout.addWidget(self.dataset_hint_label, 2, 1, alignment=Qt.AlignTop)
        dataset_layout.addLayout(icons_layout, 2, 0, 1, 3)
        dataset_group.setLayout(dataset_layout)
        self.main_layout.addWidget(dataset_group)

    def create_brain_gb(self):
        """Creates the brain controls"""
        brain_group = QGroupBox()
        brain_group.setTitle('Brain')
        brain_layout = QGridLayout()
        brain_label = QLabel('Select brain:')
        brain_label.setMaximumWidth(100)
        if self.configuration.brain_path:
            current_brain = self.configuration.brain_path.split('/')[-1].split(".")[0]   # get brain name without .py
        else:
            current_brain = ''
        self.current_brain_label = QLabel('Current brain: <b><FONT COLOR = lightgreen>' + current_brain + '</b>')
        hint_label = QLabel('(pause simulation to change brain)')
        hint_label.setStyleSheet('color: lightblue; font-size: 12px; font-style: italic')

        self.brain_combobox = QComboBox()
        self.brain_combobox.setEnabled(True)
        brains = [file.split(".")[0] for file in os.listdir(brains_path + self.configuration.robot_type)
                  if file.endswith('.py') and file.split(".")[0] != '__init__']
        self.brain_combobox.addItem('')
        self.brain_combobox.addItems(brains)
        index = self.brain_combobox.findText(current_brain, Qt.MatchFixedString)
        if index >= 0:
            self.brain_combobox.setCurrentIndex(index)
        self.brain_combobox.currentIndexChanged.connect(self.selection_change_brain)

        self.confirm_brain = QPushButton('Load')
        self.confirm_brain.setEnabled(True)
        self.confirm_brain.clicked.connect(self.load_brain)
        self.confirm_brain.setMaximumWidth(60)

        brain_layout.addWidget(brain_label, 0, 0)
        brain_layout.addWidget(self.brain_combobox, 0, 1, alignment=Qt.AlignLeft)
        brain_layout.addWidget(hint_label, 1, 1, alignment=Qt.AlignTop)
        brain_layout.addWidget(self.current_brain_label, 2, 0, 1, 2)
        brain_layout.addWidget(self.confirm_brain, 2, 1, alignment=Qt.AlignRight)

        brain_group.setLayout(brain_layout)
        self.main_layout.addWidget(brain_group)

    def create_simulation_gb(self):
        """Create the simulation controls"""
        with open(worlds_path) as f:
            data = f.read()
        worlds_dict = json.loads(data)[self.configuration.robot_type]

        sim_group = QGroupBox()
        sim_group.setTitle('Simulation')
        sim_layout = QGridLayout()

        sim_label = QLabel('Select world:')
        sim_label.setMaximumWidth(100)
        if self.configuration.current_world:
            current_world = self.configuration.current_world
        else:
            current_world = ''
        self.current_sim_label = QLabel('Current world:   <b><FONT COLOR = lightgreen>' + current_world.split('/')[-1] + '</b>')
        hint_label = QLabel('(pause simulation to change world)')
        hint_label.setStyleSheet('color: lightblue; font-size: 12px; font-style: italic')

        self.world_combobox = QComboBox()
        self.world_combobox.setEnabled(True)
        worlds = [elem['world'] for elem in worlds_dict]
        self.world_combobox.addItem('')
        self.world_combobox.addItems(worlds)
        index = self.world_combobox.findText(current_world, Qt.MatchFixedString)
        if index >= 0:
            self.world_combobox.setCurrentIndex(index)
        self.world_combobox.currentIndexChanged.connect(self.selection_change_world)

        self.confirm_world = QPushButton('Load')
        self.confirm_world.setEnabled(True)
        self.confirm_world.clicked.connect(self.load_world)
        self.confirm_world.setMaximumWidth(60)

        start_pause_simulation_label = ClickableLabel('sim', 60, QPixmap(':/assets/play.png'), parent=self)
        start_pause_simulation_label.setToolTip('Start/Pause the simulation')
        reset_simulation = ClickableLabel('reset', 40, QPixmap(':/assets/reload.png'), parent=self)
        reset_simulation.setToolTip('Reset the simulation')
        show_gzclient = ClickableLabel('gzcli', 40, QPixmap(':/assets/gazebo_light.png'), parent=self)
        show_gzclient.setToolTip('Open/Close simulator window')
        pause_reset_layout = QHBoxLayout()
        pause_reset_layout.addWidget(show_gzclient, alignment=Qt.AlignRight)
        pause_reset_layout.addWidget(start_pause_simulation_label, alignment=Qt.AlignCenter)
        pause_reset_layout.addWidget(reset_simulation, alignment=Qt.AlignLeft)

        sim_layout.addWidget(sim_label, 0, 0)
        sim_layout.addWidget(self.world_combobox, 0, 1, alignment=Qt.AlignLeft)
        sim_layout.addWidget(hint_label, 1, 1, alignment=Qt.AlignTop)
        sim_layout.addWidget(self.current_sim_label, 2, 0, 1, 2)
        sim_layout.addWidget(self.confirm_world, 2, 1, alignment=Qt.AlignRight)
        sim_layout.addLayout(pause_reset_layout, 3, 0, 1, 2)

        sim_group.setLayout(sim_layout)
        self.main_layout.addWidget(sim_group)

    def start_recording(self):
        """Callback that handles the recording initialization"""
        filename = self.file_selector_save.text()
        if os.path.isfile(filename) and filename.endswith(".bag"):
            topics = self.topics_popup.active_topics
            if len(topics) > 0:
                self.dataset_hint_label.hide()
                self.recording_animation_label.start_animation()
                self.recording_label.show()
                self.recording_animation_label.show()
                self.controller.record_rosbag(topics, self.file_selector_save.text())
            else:
                self.dataset_hint_label.setText("Select a topic to record first")
                self.dataset_hint_label.show()
                self.start_pause_record_label.active = False
                self.start_pause_record_label.setPixmap(QPixmap(':/assets/play.png'))
        else:
            self.dataset_hint_label.setText('Select a .bag file to save dataset first!')
            self.dataset_hint_label.show()
            self.start_pause_record_label.active = False
            self.start_pause_record_label.setPixmap(QPixmap(':/assets/play.png'))

    def stop_recording(self):
        """Callback that handles recording stopping"""
        self.recording_animation_label.stop_animation()
        self.recording_animation_label.hide()
        self.recording_label.hide()
        self.controller.stop_record()

    def selection_change_brain(self, i):
        # print "Items in the list are :"

        # for count in range(self.brain_combobox.count()):
        #     print self.brain_combobox.itemText(count)
        # print "Current index", i, "selection changed ", self.brain_combobox.currentText()
        pass

    def selection_change_world(self, i):

        # for count in range(self.world_combobox.count()):
        #     print self.world_combobox.itemText(count)
        # print("Current index", i, "selection changed ", self.world_combobox.currentText())
        pass

    def saveFileDialog(self):
        """Callback that will create the bag file where the dataset will be recorded"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "Bag Files (*.bag)",
                                                  options=options)
        if filename:
            if not filename.endswith(".bag"):
                filename += ".bag"
            open(filename, 'w').close()
            self.file_selector_save.setText(filename)

    def reset_simulation(self):
        """Callback that handles simulation resetting"""
        self.controller.reset_gazebo_simulation()

    def pause_simulation(self):
        """Callback that handles simulation pausing"""
        self.world_combobox.setEnabled(True)
        self.confirm_world.setEnabled(True)
        self.world_combobox.setStyleSheet('color: white')
        self.confirm_world.setStyleSheet('color: white')
        self.brain_combobox.setEnabled(True)
        self.confirm_brain.setEnabled(True)
        self.brain_combobox.setStyleSheet('color: white')
        self.confirm_brain.setStyleSheet('color: white')

        self.controller.pause_gazebo_simulation()

    def resume_simulation(self):
        """Callback that handles simulation resuming"""
        self.world_combobox.setEnabled(False)
        self.confirm_world.setEnabled(False)
        self.world_combobox.setStyleSheet('color: grey')
        self.confirm_world.setStyleSheet('color: grey')
        self.brain_combobox.setEnabled(False)
        self.confirm_brain.setEnabled(False)
        self.brain_combobox.setStyleSheet('color: grey')
        self.confirm_brain.setStyleSheet('color: grey')

        # self.controller.reload_brain(brains_path + self.brain_combobox.currentText() + '.py')
        self.controller.unpause_gazebo_simulation()

    def load_brain(self):
        """Callback that handles brain reloading"""
        brain = self.brain_combobox.currentText() + '.py'
        txt = '<b><FONT COLOR = lightgreen>' + " ".join(self.brain_combobox.currentText().split("_")) + '</b>'
        self.current_brain_label.setText('Current brain: ' + txt)
        # load brain from controller
        self.controller.reload_brain(brains_path + self.configuration.robot_type + '/' + brain)

        # save to configuration
        self.configuration.brain_path = brains_path + self.configuration.robot_type + '/' + brain

    def load_world(self):
        """Callback that handles world change"""
        world = self.world_combobox.currentText()
        txt = '<b><FONT COLOR = lightgreen>' + self.world_combobox.currentText().split('/')[-1] + '</b>'
        self.current_sim_label.setText('Current world: ' + txt)
        # Load new world
        environment.launch_env(world)
        self.controller.initialize_robot()

        # save to configuration
        self.configuration.current_world = world

    @staticmethod
    def open_close_simulator_gui():
        """Method tho enable/disable gazebo client GUI"""
        if not environment.is_gzclient_open():
            environment.open_gzclient()
        else:
            environment.close_gzclient()
