import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import ui.gui.resources.resources
from logo import Logo
from social import SocialMedia

worlds_path = '/home/fran/github/BehaviorSuite/behavior_suite/ui/gui/resources/worlds.json'
brains_path = '/home/fran/github/BehaviorSuite/behavior_suite/brains/f1/'

""" TODO:   change absolute paths
            fix dataset file selector and configuration
            initialize widget from configuration
"""


class HLine(QFrame):
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(self.HLine | self.Sunken)
        pal = self.palette()
        pal.setColor(QPalette.WindowText, QColor(255, 255, 255))
        self.setPalette(pal)


class AnimatedLabel(QLabel):

    SLOW_DURATION = 1500
    MID_DURATION = 1000
    FAST_DURATION = 500

    def __init__(self, parent=None, color='yellow'):
        QLabel.__init__(self, parent)
        self.config_animation(self.MID_DURATION)
        self.setPixmap(QPixmap(':/assets/recording.png'))
        self.setFixedSize(40,40)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: ' + color)
        self.setScaledContents(True)


    def config_animation(self, duration):
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

    def __init__(self, id, size, pmap, parent=None):
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
        # self.setStyleSheet('background-color: black')
        pass

    def leaveEvent(self, event):
        # self.setStyleSheet('background-color: rgb(0, 0, 0, 0,)')
        pass

    def mousePressEvent(self, event):
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
            elif self.id == 'load':
                self.parent.load_dataset()
            elif self.id == 'sim':
                if not self.active:
                    self.setPixmap(QPixmap(':/assets/pause.png'))
                    self.active = True
                    self.parent.resume_simulation()
                else: 
                    self.setPixmap(QPixmap(':/assets/play.png'))
                    self.active = False
                    self.parent.pause_simulation()
                    

class Toolbar(QWidget):

    def __init__(self, configuration, controller):
        super(Toolbar, self).__init__()
        # self.setStyleSheet('background-color: rgb(51,51,51); color: white;')
        self.windowsize = QSize(440, 1000)    
        self.configuration = configuration
        self.controller = controller
        self.setFixedSize(self.windowsize)
        self.initUI()

        self.setStyleSheet( """
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
        self.main_layout = QVBoxLayout()

        self.create_stats_gb()
        self.main_layout.addWidget(HLine())
        self.create_dataset_gb()
        self.main_layout.addWidget(HLine())
        self.create_brain_gb()
        self.main_layout.addWidget(HLine())
        self.create_simulation_gb()
        self.main_layout.addWidget(HLine())

        logo = Logo()
        social = SocialMedia()
        
        self.main_layout.addWidget(logo)
        self.main_layout.addWidget(social)
        self.setLayout(self.main_layout)
        

    def file_handler(self):
        print('fiel sel')

    
    def create_stats_gb(self):
        stats_group = QGroupBox()
        stats_group.setTitle('Stats')
        stats_layout = QGridLayout()

        stats_group.setLayout(stats_layout)
        self.main_layout.addWidget(stats_group)

    def create_dataset_gb(self):
        dataset_group = QGroupBox()
        dataset_group.setTitle('Dataset')
        dataset_layout = QGridLayout()
        save_path_label = QLabel('Save path:  ')
        load_path_label = QLabel('Load path:  ')
        self.file_selector_save = QLineEdit()
        self.file_selector_save.setPlaceholderText('Select dataset save path')
        self.file_selector_save.setObjectName("dataset_save")
        self.file_selector_load = QLineEdit()
        self.file_selector_load.setPlaceholderText('Select dataset load path')
        self.file_selector_load.setObjectName("dataset_load")
        selector_save_button = QPushButton('...')
        selector_save_button.setMaximumSize(30,30)
        selector_save_button.clicked.connect(lambda: self.selectFile(self.file_selector_save))
        selector_load_button = QPushButton('...')
        selector_load_button.setMaximumSize(30,30)
        selector_load_button.clicked.connect(lambda: self.selectFile(self.file_selector_save))

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum);

        icons_layout = QHBoxLayout()
        
        self.recording_animation_label = AnimatedLabel()
        self.recording_label = QLabel("Recording...")
        self.recording_label.setStyleSheet('color: yellow; font-weight: bold;')
        self.recording_animation_label.hide()
        self.recording_label.hide()
        self.recording_animation_label.setPixmap(QPixmap(':/assets/recording.png'))
        start_pause_record_label = ClickableLabel('play', 30, QPixmap(':/assets/play.png'), parent=self)
        start_pause_record_label.setToolTip('Start/Stop recording dataset')
        load_dataset_label = ClickableLabel('load', 30, QPixmap(':/assets/load.png'), parent=self)

        icons_layout.addWidget(self.recording_animation_label, alignment=Qt.AlignBottom)
        icons_layout.addWidget(self.recording_label, alignment=Qt.AlignBottom)
        icons_layout.addItem(horizontalSpacer)
        icons_layout.addWidget(load_dataset_label, alignment=Qt.AlignBottom)
        icons_layout.addWidget(start_pause_record_label, alignment=Qt.AlignBottom)

        dataset_layout.addWidget(save_path_label, 0, 0)
        dataset_layout.addWidget(self.file_selector_save, 0, 1)
        dataset_layout.addWidget(selector_save_button, 0, 2)

        dataset_layout.addWidget(load_path_label, 1, 0)
        dataset_layout.addWidget(self.file_selector_load, 1, 1)
        dataset_layout.addWidget(selector_load_button, 1, 2)

        # dataset_layout.addItem(verticalSpacer,2,0)
        dataset_layout.addLayout(icons_layout, 3, 0, 3, 3)
        dataset_group.setLayout(dataset_layout)
        self.main_layout.addWidget(dataset_group)

    def create_brain_gb(self):
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
        self.brain_combobox.setEnabled(False)
        brains = [file.split(".")[0] for file in os.listdir(brains_path) if file.endswith('.py') and file.split(".")[0] != '__init__']
        self.brain_combobox.addItem('')
        self.brain_combobox.addItems(brains)
        index = self.brain_combobox.findText(current_brain, Qt.MatchFixedString)
        if index >= 0:
            self.brain_combobox.setCurrentIndex(index)
        self.brain_combobox.currentIndexChanged.connect(self.selection_change_brain)
        self.brain_combobox.setStyleSheet("""color:gray;""")  # emulate disabled, default style doesn't work (inheritance?)

        self.confirm_brain = QPushButton('Load')
        self.confirm_brain.setEnabled(False)
        self.confirm_brain.clicked.connect(self.load_brain)
        self.confirm_brain.setMaximumWidth(60)
        self.confirm_brain.setStyleSheet('color: grey')
        
        brain_layout.addWidget(brain_label, 0, 0)
        brain_layout.addWidget(self.brain_combobox, 0, 1, alignment=Qt.AlignLeft)
        brain_layout.addWidget(hint_label,1, 1, alignment=Qt.AlignTop)
        brain_layout.addWidget(self.current_brain_label, 2, 0, 1, 2)
        brain_layout.addWidget(self.confirm_brain, 2, 1, alignment=Qt.AlignRight)

        brain_group.setLayout(brain_layout)
        self.main_layout.addWidget(brain_group)

    def create_simulation_gb(self):

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
        self.current_sim_label = QLabel('Current world:   <b><FONT COLOR = lightgreen>' + current_world+ '</b>')
        hint_label = QLabel('(pause simulation to change world)')
        hint_label.setStyleSheet('color: lightblue; font-size: 12px; font-style: italic')
        

        self.world_combobox = QComboBox()
        self.world_combobox.setEnabled(False)
        worlds = [elem['world'] for elem in worlds_dict]
        self.world_combobox.addItem('')
        self.world_combobox.addItems(worlds)
        index = self.world_combobox.findText(current_world, Qt.MatchFixedString)
        if index >= 0:
            self.world_combobox.setCurrentIndex(index)
        self.world_combobox.currentIndexChanged.connect(self.selection_change_world)
        self.world_combobox.setStyleSheet("""color:gray;""")  # emulate disabled, default style doesn't work (inheritance?)

        self.confirm_world = QPushButton('Load')
        self.confirm_world.setEnabled(False)
        self.confirm_world.clicked.connect(self.load_world)
        self.confirm_world.setMaximumWidth(60)
        self.confirm_world.setStyleSheet('color: grey')

        start_pause_simulation_label = ClickableLabel('sim', 60, QPixmap(':/assets/play.png'), parent=self)
        start_pause_simulation_label.setToolTip('Start/Pause the simulation')

        
        sim_layout.addWidget(sim_label, 0, 0)
        sim_layout.addWidget(self.world_combobox, 0, 1, alignment=Qt.AlignLeft)
        sim_layout.addWidget(hint_label,1, 1, alignment=Qt.AlignTop)
        sim_layout.addWidget(self.current_sim_label, 2, 0, 1, 2)
        sim_layout.addWidget(self.confirm_world, 2, 1, alignment=Qt.AlignRight)
        sim_layout.addWidget(start_pause_simulation_label, 3, 0, 1, 2, alignment=Qt.AlignCenter)


        sim_group.setLayout(sim_layout)
        self.main_layout.addWidget(sim_group)


    def start_recording(self):
        print('starting record')
        self.recording_animation_label.start_animation()
        self.recording_label.show()
        self.recording_animation_label.show()

    def stop_recording(self):
        print('stopping record')
        self.recording_animation_label.stop_animation()
        self.recording_animation_label.hide()
        self.recording_label.hide()
    
    def load_dataset(self):
        print('loading dataset')

    def selection_change_brain(self,i):
        print "Items in the list are :"

        for count in range(self.brain_combobox.count()):
            print self.brain_combobox.itemText(count)
        print "Current index",i,"selection changed ",self.brain_combobox.currentText()

    def selection_change_world(self,i):

        for count in range(self.world_combobox.count()):
            print self.world_combobox.itemText(count)
        print "Current index",i,"selection changed ",self.world_combobox.currentText()

    def pause_simulation(self):
        self.world_combobox.setEnabled(True)
        self.confirm_world.setEnabled(True)
        self.world_combobox.setStyleSheet('color: white')
        self.confirm_world.setStyleSheet('color: white')
        self.brain_combobox.setEnabled(True)
        self.confirm_brain.setEnabled(True)
        self.brain_combobox.setStyleSheet('color: white')
        self.confirm_brain.setStyleSheet('color: white')

        self.controller.pause_gazebo_simulation()
        self.controller.reload_brain(brains_path + self.brain_combobox.currentText() + '.py')

    def resume_simulation(self):
        self.world_combobox.setEnabled(False)
        self.confirm_world.setEnabled(False)
        self.world_combobox.setStyleSheet('color: grey')
        self.confirm_world.setStyleSheet('color: grey')
        self.brain_combobox.setEnabled(False)
        self.confirm_brain.setEnabled(False)
        self.brain_combobox.setStyleSheet('color: grey')
        self.confirm_brain.setStyleSheet('color: grey')

        self.controller.unpause_gazebo_simulation()

    def load_brain(self):
        brain = self.brain_combobox.currentText() + '.py'
        self.current_brain_label.setText('Current brain:   <b><FONT COLOR = lightgreen>' + " ".join(self.brain_combobox.currentText().split("_")) + '</b>')
        # load brain from controller
        self.controller.reload_brain(brains_path + brain)

        #save to configuration
        self.configuration.brain_path = brains_path + brain
        print('current brain', brains_path + brain)

    def load_world(self):
        world = self.world_combobox.currentText()
        self.current_sim_label.setText('Current world:   <b><FONT COLOR = lightgreen>' + self.world_combobox.currentText() + '</b>')
        # load brain from controller

        #save to configuration
        self.configuration.current_world = world
        print('current world', world)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = Toolbar()
    ex.show()

    sys.exit(app.exec_())
