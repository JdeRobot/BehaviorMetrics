#!/usr/bin/env python
""" This module is responsible for the layout of frames view.

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

from collections import defaultdict

from PyQt5.QtCore import (QPropertyAnimation, QSequentialAnimationGroup, Qt,
                          pyqtSignal)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout, QWidget)

from ui.gui.views.logo import Logo

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

current_selection_id = 0


class InfoLabel(QLabel):
    """This is a helper class that adds a tooltip to a default QLabel."""

    def __init__(self, description='Click me!', parent=None):
        """ Constructor of the class

        Arguments:
            description {str} -- Text for the tooltip
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget
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
        """ Mouse event when hover the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(60, 60, Qt.KeepAspectRatio))

    def leaveEvent(self, event):
        """Mouse event when exit the widget"""
        self.setPixmap(QPixmap(':/assets/info_icon.png').scaled(50, 50, Qt.KeepAspectRatio))

    def mousePressEvent(self, event):
        """Mouse event when press the widget"""
        if event.button() & Qt.LeftButton:
            self.parent.show_information_popup()


class AnimatedLabel(QLabel):
    """This is a helper class that extends the functioality of the default QLabel with a fade in-out animation"""

    SLOW_DURATION = 1500
    FAST_DURATION = 500

    def __init__(self, parent=None, color='yellow'):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views.main_view.MainView} -- parent of this widget (default: {None})
            color {str} -- Foreground color of the label (default: {'yellow'})
        """
        QLabel.__init__(self, parent)
        self.start_animation(self.SLOW_DURATION)
        font = QFont('Arial', 30)
        self.setFont(font)
        self.setText("Select your layout")
        self.setFixedHeight(100)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('color: ' + color)

    def start_animation(self, duration):
        """Start the fading animation of the label

        Arguments:
            duration {int} -- Duration in milliseconds of a complete fade in-fade out cycle
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


class ClickableQFrame(QFrame):
    """This class extends the QFrame and with mouse events"""

    # default style
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

    # hover style
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
        """Constructor of the class

        Arguments:
            parent {ui.gui.views.main_view.MainView} -- Parent of this widget
            row {int} -- Position of the frame in rows
            col {int} -- Position of the frame in columns
        """
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
        """Mouse event when hover frame"""
        if not self.is_selected:
            self.setStyleSheet(self.hover_qss)

    def leaveEvent(self, event):
        """Mouse event when leave frame"""
        if not self.is_selected:
            self.setStyleSheet(self.normal_qss)

    def mousePressEvent(self, event):
        """Mouse event when pressed on widget"""
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
        """Getter for the frame position in row, column

        Returns:
            tuple -- Position of the frame in the main window (row, column)
        """
        return (self.row, self.col)

    def clear(self):
        """Clear the current selected frame from selected ones"""
        self.is_selected = False
        self.group_id = -1
        self.labelgr.setText('')
        self.setStyleSheet(self.normal_qss)


class FakeToolbar(QWidget):
    """ This is a helper class for previsualization purposes"""

    def __init__(self, parent=None):
        """ Constructor of the class"""
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
    """Class that create the matrix view of the frames.

    This class is the responsible for painting the matrix of the different frames to be selected.
    This matrix will be used to setup the main window view with the user selection."""

    ROWS = 3
    COLS = 3

    def __init__(self, parent=None):
        """Constructor of the class

        Keyword Arguments:
            parent {ui.gui.views.main_view.MainView} -- parent of this widget (default: {None})
        """
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
        """Retrieve info of the final layout"""
        groups = defaultdict(list)
        for frame in self.frames:
            groups[frame.group_id].append(frame)
        return groups

    def clear_frames(self):
        """Clear all the selections made from the user"""
        for frame in self.frames:
            frame.clear()


class LayoutSelection(QWidget):
    """Main class of the layout selector view

    This class is responsible to paint the layout selector of the application and the controls to modify it."""

    switch_window = pyqtSignal()

    def __init__(self, configuration, parent=None):
        """Constructor of the class

        Arguments:
            configuration {utils.configuration.Config} -- Configuration of the application

        Keyword Arguments:
            parent {ui.gui.views_controller.ParentWindow} -- Parent of this widget (default: {None})
        """
        super(LayoutSelection, self).__init__(parent)
        self.parent = parent
        self.configuration = configuration
        self.parent.status_bar.showMessage('LMB for single selection ---- Ctrl + LMB for multiple selection')
        self.initUI()

    def initUI(self):
        """Initialize the GUI elements"""

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
        """Confirm the selection of the configuration of the layouts"""
        self.switch_window.emit()
        self.configuration.create_layout_from_gui(self.get_config())

    def get_config(self):
        """Retrieve the configuration of the layout"""
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
            gs.append((min_row, min_col, (max_row - min_row)+1, (max_col - min_col)+1, id))
        return gs

    def preview_win(self, ):
        """Preview the selection as it will be in the main window"""
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
        """Clear selection"""
        global current_selection_id
        current_selection_id = 0
        self.base.clear_frames()

    def update_gui(self):
        pass

    def show_information_popup(self):
        pass
