from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QWidget, QMainWindow)


class StatsWindow(QMainWindow):
    def __init__(self, parent=None, controller=None):
        super(StatsWindow, self).__init__(parent)
        
        self.controller = controller
        self.setWindowTitle("Statistics")        
        wid = QWidget(self)
        self.setCentralWidget(wid)
        
        self.layout = QVBoxLayout()
        self.percentage_completed_label = QLabel("Percentage completed -> " + str(self.controller.lap_statistics['percentage_completed']) + "%")
        self.layout.addWidget(self.percentage_completed_label)
        self.completed_distance_label = QLabel("Completed distance -> " + str(self.controller.lap_statistics['completed_distance']) + "m")
        self.layout.addWidget(self.completed_distance_label)
        self.orientation_mae_label = QLabel("Orientation MAE -> " + str(self.controller.lap_statistics['orientation_mae']))
        self.layout.addWidget(self.orientation_mae_label)
        self.orientation_total_err_label = QLabel("Orientation total error -> " + str(self.controller.lap_statistics['orientation_total_err']))
        self.layout.addWidget(self.orientation_total_err_label)
        
        # If lap is completed, extend information
        if 'lap_seconds' in self.controller.lap_statistics:
            self.lap_seconds_label = QLabel("Lap seconds -> " + str(self.controller.lap_statistics['lap_seconds']) + "s")
            self.layout.addWidget(self.lap_seconds_label)
            self.average_speed_label = QLabel("Average speed -> " + str(self.controller.lap_statistics['average_speed']) + "m/s")
            self.layout.addWidget(self.average_speed_label)
            self.circuit_diameter_label = QLabel("Circuit diameter -> " + str(self.controller.lap_statistics['circuit_diameter']) + "m")
            self.layout.addWidget(self.circuit_diameter_label)
            
        wid.setLayout(self.layout)
        