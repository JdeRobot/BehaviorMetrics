import threading


class Controller:

    def __init__(self):
        pass
        self.data_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.data = {}
        self.pose3D_data = None

    def update_frame(self, frame_id, data):
        with self.data_lock:
            self.data[frame_id] = data
            # print(self.data[frame_id])

    def get_data(self, frame_id):
        with self.data_lock:
            data = self.data[frame_id]
            # self.data[frame_id] = None

        return data

    def update_pose3d(self, data):
        with self.pose_lock:
            self.pose3D_data = data

    def get_pose3D(self):
        return self.pose3D_data
