from collections import defaultdict

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

    def __init__(self, parent, row, col):
        QFrame.__init__(self)
        self.setObjectName('customframe')
        self.parent = parent
        self.is_selected = False
        self.group_id = -1
        self.row = row
        self.col = col
        self.setStyleSheet(self.normal_qss)
        lay = QVBoxLayout()
        global current_selection_id
        self.labelgr = QLabel()
        self.labelgr.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.labelgr)
        self.setLayout(lay)

    def enterEvent(self, event):
        if not self.is_selected:
            self.setStyleSheet(self.hover_qss)

    def leaveEvent(self, event):
        if not self.is_selected:
            self.setStyleSheet(self.normal_qss)

    def mousePressEvent(self, event):
        global current_selection_id
        if event.buttons() & Qt.LeftButton:
            if not self.is_selected:
                modifiers = QApplication.keyboardModifiers()
                if not modifiers == Qt.ControlModifier:
                    current_selection_id += 1
                self.is_selected = True
                self.group_id = current_selection_id
                self.labelgr.setText(str(self.group_id))
            else:
                self.is_selected = False
                self.group_id = -1
                self.labelgr.setText('')

    def get_position(self):
        return (self.row, self.col)

    def clear(self):
        self.is_selected = False
        self.group_id = -1
        self.labelgr.setText('')
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
        self.main_layout.addWidget(b1)
        self.setLayout(self.main_layout)


class LayoutMatrix(QWidget):

    ROWS = 3
    COLS = 3

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.dragstart = None
        self.main_layout = QGridLayout()

        # add matrix frame
        self.frames = []
        for row in range(self.ROWS):
            for col in range(1, self.COLS + 1):   # +1 because toolbar is placed in first column
                frame = ClickableQFrame(self, row, col)
                frame.setMouseTracking(True)
                self.frames.append(frame)
                self.main_layout.addWidget(frame, row, col)
        self.setLayout(self.main_layout)

    def collect_groups(self):
        """ retrieve info of the final layout"""
        groups = defaultdict(list)
        for frame in self.frames:
            groups[frame.group_id].append(frame)
        return groups

    def clear_frames(self):
        for frame in self.frames:
            frame.clear()


class LayoutSelection(QWidget):
    updGUI = pyqtSignal()
    switch_window = pyqtSignal()

    def __init__(self, parent=None):
        super(LayoutSelection, self).__init__(parent)
        self.updGUI.connect(self.update_gui)
        self.parent = parent
        self.parent.status_bar.showMessage('LMB for single selection ---- Ctrl + LMB for multiple selection')
        self.initUI()

    def initUI(self):

        main_layout = QVBoxLayout()
        self.setStyleSheet('background-color: rgb(51,51,51); color: white')
        self.setLayout(main_layout)

        # define view's widgets
        logo = Logo()
        self.base = LayoutMatrix()

        confirm_button = QPushButton("Confirm", self)
        confirm_button.setFixedSize(100, 50)
        confirm_button.clicked.connect(self.confirm)

        preview_button = QPushButton("Preview", self)
        preview_button.setFixedSize(100, 50)
        preview_button.clicked.connect(self.preview_win)

        clear_button = QPushButton("Clear", self)
        clear_button.setFixedSize(100, 50)
        clear_button.clicked.connect(self.clear)

        lbl = AnimatedLabel(self)

        # insert widgets in layouts
        butons_layout = QHBoxLayout()
        butons_layout.addWidget(clear_button)
        butons_layout.addWidget(preview_button)
        butons_layout.addWidget(confirm_button)
        butons_layout.addWidget(InfoLabel(parent=self))

        main_layout.addWidget(logo)
        main_layout.addWidget(self.base)
        main_layout.addLayout(butons_layout)
        main_layout.addWidget(lbl)

        self.show_information_popup()

    def confirm(self):
        self.switch_window.emit()

    def get_config(self):
        groups = self.base.collect_groups()
        gs = []
        for id in groups:
            if id == -1:
                continue
            frames = groups[id]
            min_row = min(frame.row for frame in frames)
            min_col = min(frame.col for frame in frames)
            max_row = max(frame.row for frame in frames)
            max_col = max(frame.col for frame in frames)
            # print('(widget, {}, {}, {}, {})'.format(min_row, min_col, (max_row - min_row)+1, (max_col - min_col)+1))
            gs.append((min_row, min_col, (max_row - min_row)+1, (max_col - min_col)+1))
        return gs

    def preview_win(self, ):
        self.preview = QWidget()
        self.preview.setStyleSheet('background-color: rgb(51,51,51)')
        self.preview.setFixedSize(1750, 900)

        greedy = QGridLayout()
        self.preview.setLayout(greedy)

        positions = self.get_config()
        for c in positions:
            lbl = QLabel()
            lbl.setStyleSheet('border: 2px solid white')
            lbl.setPixmap(QPixmap(':/assets/logo_200.svg'))
            lbl.setAlignment(Qt.AlignCenter)
            greedy.addWidget(lbl, c[0], c[1], c[2], c[3])
        greedy.addWidget(FakeToolbar(), 0, 0)

        self.preview.show()

    def clear(self):
        global current_selection_id
        current_selection_id = 0
        self.base.clear_frames()

    def update_gui(self):
        pass

    def show_information_popup(self):
        pass
