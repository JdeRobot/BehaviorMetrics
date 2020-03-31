import os
import sys
from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import ui.gui.resources.resources

from social import SocialMedia
from logo import Logo


brains_path = '/home/fran/github/BehaviorSuite/behavior_suite/brains/f1/'

class AnimatedLabel(QLabel):

    SLOW_DURATION = 1500
    MID_DURATION = 1000
    FAST_DURATION = 500

    def __init__(self, parent=None, color='yellow'):
        QLabel.__init__(self, parent)
        self.start_animation(self.MID_DURATION)
        self.setPixmap(QPixmap(':/assets/recording.png'))
        self.setFixedSize(40,40)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: ' + color)
        self.setScaledContents(True)


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
        self.ga.addAnimation(self.animation2)
        self.ga.addAnimation(self.animation1)
        self.ga.setLoopCount(-1)
        self.ga.start()

class ClickableLabel(QLabel):

    def __init__(self, id, pmap, parent=None):
        QLabel.__init__(self, parent)
        self.setMaximumSize(30, 30)
        self.parent = parent
        self.setStyleSheet('background-color: rgba(0, 0, 0, 0)')
        self.setPixmap(pmap)
        self.setScaledContents(True)
        self.resize(20,20)
        self.id = id
        self.active = False

    def enterEvent(self, event):
        self.setStyleSheet('background-color: black')

    def leaveEvent(self, event):
        self.setStyleSheet('background-color: rgb(0, 0, 0, 0,)')

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

class ExampleWindow(QWidget):

    def __init__(self, sensors):
        super(ExampleWindow, self).__init__()
        # self.setStyleSheet('background-color: rgb(51,51,51); color: white;')
        self.windowsize = QSize(500, 900)    
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
        self.create_dataset_gb()
        self.create_brain_gb()
        self.create_simulation_gb()

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
        start_pause_record_label = ClickableLabel('play', QPixmap(':/assets/play.png'), parent=self)
        load_dataset_label = ClickableLabel('load', QPixmap(':/assets/load.png'), parent=self)

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
        current_brain_label = QLabel('Current brain:   <b><FONT COLOR = lightgreen>' + " ".join('brain_f1_opencv'.split("_")) + '</b>')
        hint_label = QLabel('(pause simulation to change brain)')
        hint_label.setStyleSheet('color: lightblue; font-size: 12px; font-style: italic')

        self.cb = QComboBox()
        self.cb.setEnabled(False)
        brains = [file.split(".")[0] for file in os.listdir(brains_path) if file.endswith('.py') and file.split(".")[0] != '__init__']
        self.cb.addItems(brains)
        self.cb.currentIndexChanged.connect(self.selectionchange)
        self.cb.setStyleSheet("""color:gray;""")  # emulate disabled, default style doesn't work (inheritance?)
        
        
        brain_layout.addWidget(brain_label, 0, 0)
        brain_layout.addWidget(self.cb, 0, 1)
        brain_layout.addWidget(hint_label,1, 1, alignment=Qt.AlignTop)
        brain_layout.addWidget(current_brain_label, 2, 0, 1, 2)

 

        brain_group.setLayout(brain_layout)
        self.main_layout.addWidget(brain_group)

    def create_simulation_gb(self):
        sim_group = QGroupBox()
        sim_group.setTitle('Simulation')
        sim_layout = QGridLayout()

        sim_group.setLayout(sim_layout)
        self.main_layout.addWidget(sim_group)

    def start_recording(self):
        print('starting record')
        self.recording_label.show()
        self.recording_animation_label.show()

    def stop_recording(self):
        print('stopping record')
        self.recording_animation_label.hide()
        self.recording_label.hide()
    
    def load_dataset(self):
        print('loading dataset')

    def selectionchange(self,i):
        print "Items in the list are :"

        for count in range(self.cb.count()):
            print self.cb.itemText(count)
        print "Current index",i,"selection changed ",self.cb.currentText()

    def pause_simulation(self):
        self.cb.setEnabled(True)
        self.cb.setStyleSheet('color: white')

    def resume_simulation(self):
        self.cb.setEnabled(False)
        self.cb.setStyleSheet('color: grey')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    screensize = QSize(800,600)

    ex = ExampleWindow(screensize)
    ex.show()

    sys.exit(app.exec_())
