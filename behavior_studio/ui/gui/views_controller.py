#!/usr/bin/env python
""" This module contains the handler of the different views of the application.

This module will handle the order in which the different views are shown, and pass all the information
within them.

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

import datetime
import sys
import time

from PyQt5.QtCore import QPropertyAnimation, QSize, QTimer, pyqtSignal
# from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (QApplication, QDesktopWidget, QFrame,
                             QGraphicsOpacityEffect, QLabel, QMainWindow,
                             QVBoxLayout, QWidget)

from ui.gui.threadGUI import ThreadGUI
from ui.gui.views.layout_selection import LayoutSelection
from ui.gui.views.main_view import MainView
from ui.gui.views.title import TitleWindow

__author__ = 'fqez'
__contributors__ = []
__license__ = 'GPLv3'

WIDTH = 1500
HEIGHT = 1000


class VLine(QFrame):
    """Helper class that creates a vertical separator"""
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class ParentWindow(QMainWindow):
    """Main window of the application.

    This window will contain all the views managed by the ViewsController class. Works like a views container.
    """

    def __init__(self):
        """Constructor of the class"""
        super(ParentWindow, self).__init__()
        self.windowsize = QSize(WIDTH, HEIGHT)
        self.initUI()
        self.robot_selection = None
        self.closing = False

    def location_on_the_screen(self):
        """Helper function that will locate the window in a specific position in the desktop"""
        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()

        widget = self.geometry()
        x = ag.width() - widget.width()
        y = 2 * ag.height() - sg.height() - widget.height()
        self.move(x, y)

    def initUI(self):
        """Initialize all the GUI elements"""
        # self.setFixedSize(self.windowsize)
        self.setMinimumSize(self.windowsize)
        self.init_statusbar()

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        self.main_layout = QVBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet('background-color: rgb(51,51,51)')
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        self.location_on_the_screen()

    def init_statusbar(self):
        """Create and fill the bottom status bar of the application including hint messages and a clock"""

        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet('background-color: #444444; color: white; QStatusBar::item {border: none;}')

        self.status_title = QLabel()
        self.status_title.setStyleSheet('border: 0; color:  white;')
        self.status_date = QLabel()
        self.status_date.setStyleSheet('border: 0; color:  white;')

        self.status_bar.addPermanentWidget(VLine())
        self.status_bar.addPermanentWidget(self.status_title)
        self.status_bar.addPermanentWidget(VLine())
        self.status_bar.addPermanentWidget(self.status_date)
        self.status_bar.addPermanentWidget(VLine())

        self.status_title.setText("Behavior Suite ")

    #     self.animation_label = QLabel()
    #     self.animation_label.setFixedSize(20, 20)
    #     self.animation_label.setScaledContents(True)
    #     self.movie = QMovie("/home/fran/load.gif")
    #     self.animation_label.setMovie(self.movie)
    #     self.status_bar.addPermanentWidget(self.animation_label)

    # def start_load_animation(self):
    #     # self.animation_label.show()
    #     self.movie.start()
    #     self.update()

    # def stop_load_animation(self):
    #     # self.animation_label.hide()
    #     self.movie.stop()

    def recurring_timer(self):
        """Handles the statusbar clock update"""
        hour = datetime.datetime.now()
        dt_string = hour.strftime("%d/%m/%Y %H:%M:%S")
        self.status_date.setText(dt_string)

    def update_gui(self):
        pass

    def closeEvent(self, event):
        """Helper function to safe kill the application without segments violation"""
        self.closing = True
        time.sleep(0.2)
        event.accept()


class ViewsController(QMainWindow):
    """This class will handle all the views of the application.

    Responsible for showing the differnet views in a specific order depending on the input of the user. If the
    application is launched with a profile (configuration file), the main view of the application will be shown;
    otherwise, the title and the configuration views will be shown prior to the main view.
    """

    home_singal = pyqtSignal()
    robot_select_signal = pyqtSignal()

    def __init__(self, parent, configuration, controller=None):
        """Constructor of the class.

        Arguments:
            parent {ui.gui.views_controller.ParentWindow} -- Parent of this.
            configuration {utils.configuration.Config} -- Configuration instance of the application

        Keyword Arguments:
            controller {utils.controller.Controller} -- Controller of the application (default: {None})
        """
        QMainWindow.__init__(self)
        self.parent = parent
        self.controller = controller
        self.configuration = configuration
        self.main_view = None
        self.thread_gui = ThreadGUI(self)
        self.thread_gui.daemon = True

        # self.home_singal.connect(self.show_title)
        # self.robot_select_signal.connect(self.show_robot_selection)

    def show_title(self):
        """Shows the title view"""
        title = TitleWindow(self.parent)
        title.switch_window.connect(self.show_robot_selection)
        self.parent.main_layout.addWidget(title)
        self.fadein_animation()

    def show_robot_selection(self):
        """Shows the robot selection view"""
        from ui.gui.views.robot_selection import RobotSelection

        delete_widgets_from(self.parent.main_layout)
        robot_selector = RobotSelection(self.parent)
        robot_selector.switch_window.connect(self.show_world_selection)
        self.parent.main_layout.addWidget(robot_selector, 0)
        self.fadein_animation()

    def show_world_selection(self):
        """Shows the world selection view"""
        from ui.gui.views.world_selection import WorldSelection

        delete_widgets_from(self.parent.main_layout)
        world_selector = WorldSelection(self.parent.robot_selection, self.configuration, self.parent)
        world_selector.switch_window.connect(self.show_layout_selection)
        self.parent.main_layout.addWidget(world_selector)
        self.fadein_animation()

    def show_layout_selection(self):
        """Show the layout configuration view"""
        delete_widgets_from(self.parent.main_layout)
        self.layout_selector = LayoutSelection(self.configuration, self.parent)
        self.layout_selector.switch_window.connect(self.show_main_view_proxy)
        self.parent.main_layout.addWidget(self.layout_selector)
        self.fadein_animation()

    def show_main_view_proxy(self):
        """Helper function to show the main view. Will close the parent window to create  a new one"""
        # self.show_main_view(False)
        self.parent.close()

    def show_main_view(self, from_main):
        """Shows the main window depending on where the application comes from.

        If the from_main flag is true, the configuration comes from the previous GUI views. Otherwise, the configuration
        comes from a configuration file. Eitherway, the main view will be shown with the proper configuration.

        Arguments:
            from_main {bool} -- tells if the configuration comes from either configuration file or GUI.
        """
        if not from_main:
            layout_configuration = self.layout_selector.get_config()
            delete_widgets_from(self.parent.main_layout)
        else:
            layout_configuration = None
        self.main_view = MainView(layout_configuration, self.configuration, self.controller, self.parent)
        self.parent.main_layout.addWidget(self.main_view)
        self.fadein_animation()
        self.start_thread()

    def start_thread(self):
        """Start the GUI refresing loop"""
        self.thread_gui.start()

    def fadein_animation(self):
        """Start a fadein animation for views transitions"""
        self.w = QFrame(self.parent)
        # self.parent.main_layout.addWidget(self.w, 0)
        self.w.setFixedSize(WIDTH, HEIGHT)
        self.w.setStyleSheet('background-color: rgba(51,51,51,1)')
        self.w.show()

        effect = QGraphicsOpacityEffect()
        self.w.setGraphicsEffect(effect)

        self.animation = QPropertyAnimation(effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)

        self.animation.start(QPropertyAnimation.DeleteWhenStopped)
        self.animation.finished.connect(self.fade_animation)

    def fade_animation(self):
        """Safe kill the animation"""
        self.w.close()
        del self.w
        del self.animation

    def update_gui(self):
        """Update the GUI. Called from the refresing loop thread"""
        while not self.parent.closing:
            if self.main_view:
                self.main_view.update_gui()
            time.sleep(0.1)


def delete_widgets_from(layout):
    """ memory secure deletion of widgets (does not work always...)
    TODO: review this function"""
    for i in reversed(range(layout.count())):
        widgetToRemove = layout.itemAt(i).widget()
        # remove it from the layout list
        layout.removeWidget(widgetToRemove)
        # remove it from the gui
        widgetToRemove.setParent(None)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = ParentWindow()
    main_window.show()

    views_controller = ViewsController(main_window)
    views_controller.show_title()

    # th = ThreadGUI(main_window)
    # th.daemon = True
    # th.start()

    sys.exit(app.exec_())
