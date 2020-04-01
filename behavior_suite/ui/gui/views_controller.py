import datetime
import sys
import time

from PyQt5.QtCore import QPropertyAnimation, QSize, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QFrame, QGraphicsOpacityEffect,
                             QLabel, QMainWindow, QVBoxLayout, QWidget)

from ui.gui.threadGUI import ThreadGUI
from views.layout_selection import LayoutSelection
from views.main_view import MainView
from views.robot_selection import RobotSelection
from views.title import TitleWindow
from views.world_selection import WorldSelection

WIDTH = 1700
HEIGHT = 1000


class VLine(QFrame):
    # a simple VLine, like the one you get from designer
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine | self.Sunken)


class ParentWindow(QMainWindow):

    def __init__(self):
        super(ParentWindow, self).__init__()
        self.windowsize = QSize(WIDTH, HEIGHT)
        self.initUI()

        self.robot_selection = None

    def initUI(self):
        self.setFixedSize(self.windowsize)
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

    def init_statusbar(self):
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

    def recurring_timer(self):
        hour = datetime.datetime.now()
        dt_string = hour.strftime("%d/%m/%Y %H:%M:%S")
        self.status_date.setText(dt_string)

    def update_gui(self):
        pass


class ViewsController(QMainWindow):

    home_singal = pyqtSignal()
    robot_select_signal = pyqtSignal()

    def __init__(self, parent, controller, configuration):
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
        title = TitleWindow(self.parent)
        title.switch_window.connect(self.show_robot_selection)
        self.parent.main_layout.addWidget(title)
        self.fadein_animation()

    def show_robot_selection(self):
        delete_widgets_from(self.parent.main_layout)
        robot_selector = RobotSelection(self.parent)
        robot_selector.switch_window.connect(self.show_world_selection)
        self.parent.main_layout.addWidget(robot_selector, 0)
        self.fadein_animation()

    def show_world_selection(self):
        delete_widgets_from(self.parent.main_layout)
        world_selector = WorldSelection(self.parent.robot_selection, self.configuration, self.parent)
        world_selector.switch_window.connect(self.show_layout_selection)
        self.parent.main_layout.addWidget(world_selector)
        self.fadein_animation()

    def show_layout_selection(self):
        delete_widgets_from(self.parent.main_layout)
        self.layout_selector = LayoutSelection(self.configuration, self.parent)
        self.layout_selector.switch_window.connect(self.show_main_view)
        self.parent.main_layout.addWidget(self.layout_selector)
        self.fadein_animation()

    def show_main_view(self):
        layout_configuration = self.layout_selector.get_config()
        delete_widgets_from(self.parent.main_layout)
        self.main_view = MainView(layout_configuration, self.configuration, self.controller, self.parent)
        self.parent.main_layout.addWidget(self.main_view)
        self.fadein_animation()
        self.start_thread()

    def start_thread(self):
        self.thread_gui.start()

    def fadein_animation(self):
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
        self.animation.finished.connect(self.fff)

    def fff(self):
        self.w.close()
        del self.w
        del self.animation

    def update_gui(self):
        while 1:
            if self.main_view:
                self.main_view.update_gui()
            time.sleep(0.1)


def delete_widgets_from(layout):
    """ memory secure. """
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
