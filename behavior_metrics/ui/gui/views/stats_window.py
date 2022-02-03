from PyQt5.QtWidgets import (QLabel, QVBoxLayout, QWidget, QMainWindow)


class StatsWindow(QMainWindow):
    def __init__(self, parent=None, controller=None):
        super(StatsWindow, self).__init__(parent)

        self.controller = controller
        self.setWindowTitle("Metrics")
        wid = QWidget(self)
        self.setCentralWidget(wid)

        self.layout = QVBoxLayout()
        self.percentage_completed_label = QLabel("Percentage completed -> " + str(self.controller.experiment_metrics['percentage_completed']) + "%")
        self.layout.addWidget(self.percentage_completed_label)
        self.completed_distance_label = QLabel("Completed distance -> " + str(self.controller.experiment_metrics['completed_distance']) + "m")
        self.layout.addWidget(self.completed_distance_label)
        self.average_speed_label = QLabel("Average speed -> " + str(self.controller.experiment_metrics['average_speed']) + "m/s")
        self.layout.addWidget(self.average_speed_label)
        self.position_deviation_mae_label = QLabel("Position deviation MAE -> " + str(self.controller.experiment_metrics['position_deviation_mae']))
        self.layout.addWidget(self.position_deviation_mae_label)
        self.position_deviation_total_err_label = QLabel("Position deviation total error -> " + str(self.controller.experiment_metrics['position_deviation_total_err']))
        self.layout.addWidget(self.position_deviation_total_err_label)
        self.mean_brain_iteration_real_time_label = QLabel("Mean brain iteration real time -> " + str(self.controller.experiment_metrics['mean_brain_iteration_real_time']) + 's')
        self.layout.addWidget(self.mean_brain_iteration_real_time_label)
        self.brain_iterations_frequency_real_time_label = QLabel("Brain iterations frequency real time-> " + str(self.controller.experiment_metrics['brain_iterations_frequency_real_time']) + 'it/s')
        self.layout.addWidget(self.brain_iterations_frequency_real_time_label)
        self.target_brain_iteration_real_time_label = QLabel("Target brain iteration real time -> " + str(self.controller.experiment_metrics['target_brain_iteration_real_time']) + 'it/s')
        self.layout.addWidget(self.target_brain_iteration_real_time_label)

        self.brain_iterations_frequency_simulated_time_label = QLabel("Brain iterations frequency simulated time-> " + str(self.controller.experiment_metrics['brain_iterations_frequency_simulated_time']) + 'it/s')
        self.layout.addWidget(self.brain_iterations_frequency_simulated_time_label)
        self.target_brain_iteration_simulated_time_label = QLabel("Target brain iteration simulated time -> " + str(self.controller.experiment_metrics['target_brain_iteration_simulated_time']) + 'it/s')
        self.layout.addWidget(self.target_brain_iteration_simulated_time_label)

        self.mean_inference_time_label = QLabel("Mean inference time -> " + str(self.controller.experiment_metrics['mean_inference_time']) + 's')
        self.layout.addWidget(self.mean_inference_time_label)
        self.frame_rate_label = QLabel("Frame rate -> " + str(self.controller.experiment_metrics['frame_rate']) + 'fps')
        self.layout.addWidget(self.frame_rate_label)
        self.gpu_inference_label = QLabel("GPU inference -> " + str(self.controller.experiment_metrics['gpu_inference']))
        self.layout.addWidget(self.gpu_inference_label)
        self.mean_brain_iteration_simulated_time_label = QLabel("Mean brain iteration simulated time -> " + str(self.controller.experiment_metrics['mean_brain_iteration_simulated_time']))
        self.layout.addWidget(self.mean_brain_iteration_simulated_time_label)
        self.real_time_factor_label = QLabel("Real time factor -> " + str(self.controller.experiment_metrics['real_time_factor']))
        self.layout.addWidget(self.real_time_factor_label)
        self.real_time_update_rate_label = QLabel("Real time update rate -> " + str(self.controller.experiment_metrics['real_time_update_rate']))
        self.layout.addWidget(self.real_time_update_rate_label)
        self.experiment_total_simulated_time_label = QLabel("Experiment total simulated time -> " + str(self.controller.experiment_metrics['experiment_total_simulated_time']) + 's')
        self.layout.addWidget(self.experiment_total_simulated_time_label)
        self.experiment_total_real_time_label = QLabel("Experiment total real time -> " + str(self.controller.experiment_metrics['experiment_total_real_time']) + 's')
        self.layout.addWidget(self.experiment_total_real_time_label)

        # If lap is completed, extend information
        if 'lap_seconds' in self.controller.experiment_metrics:
            self.lap_seconds_label = QLabel(
                "Lap seconds -> " + str(self.controller.experiment_metrics['lap_seconds']) + "s")
            self.layout.addWidget(self.lap_seconds_label)
            self.circuit_diameter_label = QLabel(
                "Circuit diameter -> " + str(self.controller.experiment_metrics['circuit_diameter']) + "m")
            self.layout.addWidget(self.circuit_diameter_label)

        wid.setLayout(self.layout)
