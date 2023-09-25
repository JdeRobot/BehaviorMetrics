from pydantic import BaseModel

from ..image_f1 import Image, ListenerCamera
from .images import F1GazeboImages
from .utils import F1GazeboUtils
from .rewards import F1GazeboRewards
from .simplified_perception import (
    F1GazeboSimplifiedPerception,
)


class F1GazeboTFConfig(BaseModel):
    def __init__(self, **config):
        self.simplifiedperception = F1GazeboSimplifiedPerception()
        self.f1gazeborewards = F1GazeboRewards()
        self.f1gazeboutils = F1GazeboUtils()
        self.f1gazeboimages = F1GazeboImages()

        self.image = Image()
        #self.image = ListenerCamera("/F1ROS/cameraL/image_raw")
        self.image_raw_from_topic = None
        self.f1_image_camera = None
        self.sensor = config["sensor"]

        # Image
        self.image_resizing = config["image_resizing"] / 100
        self.new_image_size = config["new_image_size"]
        self.raw_image = config["raw_image"]
        self.height = int(config["height_image"] * self.image_resizing)
        self.width = int(config["width_image"] * self.image_resizing)
        self.center_image = int(config["center_image"] * self.image_resizing)
        self.num_regions = config["num_regions"]
        self.pixel_region = int(self.center_image / self.num_regions) * 2
        self.telemetry_mask = config["telemetry_mask"]
        self.poi = config["x_row"][0]
        self.image_center = None
        self.right_lane_center_image = config["center_image"] + (
            config["center_image"] // 2
        )
        self.lower_limit = config["lower_limit"]

        # States
        self.state_space = config["states"]
        if self.state_space == "spn":
            self.x_row = [i for i in range(1, int(self.height / 2) - 1)]
        else:
            self.x_row = config["x_row"]

        # Actions
        self.action_space = config["action_space"]
        self.actions = config["actions"]

        # Rewards
        self.reward_function = config["reward_function"]
        self.rewards = config["rewards"]
        self.min_reward = config["min_reward"]

        # Others
        self.telemetry = config["telemetry"]
