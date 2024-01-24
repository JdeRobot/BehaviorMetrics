
import tensorflow as tf
from gym.envs.registration import register
from brains.gazebo.f1.rl_utils.inference import InferencerWrapper
import yaml
import gym
import numpy as np
import time

# F1 envs
if 'F1Env-v0' not in gym.envs.registry.env_specs:
    gym.envs.register(
        id='F1Env-v0',
        entry_point='brains.gazebo.f1.rl_utils.models:F1Env',
        # More arguments here
    )
else:
    print("Environment F1Env-v0 is already registered.")




class LoadEnvVariablesDQNGazebo:
    """
    ONLY FOR DQN algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        # self.environment["gazebo_start_pose"] = [
        #     config[self.environment_set][self.env]["circuit_positions_set"][0]
        # ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agent"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agent"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agent"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agent"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agent"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agent"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agent"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agent"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["model_name"] = config["algorithm"]["dqn"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]


# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from pydantic import BaseModel
class InferenceExecutorValidator(BaseModel):
    settings: dict
    agent: dict
    environment: dict
    algorithm: dict
    inference: dict
    # gazebo: dict


class Brain:
    def __init__(self, sensors, actuators, handler, config=None):
        self.camera = sensors.get_camera('camera_0')
        self.motors = actuators.get_motor('motors_0')
        self.handler = handler
        self.config = config
        self.suddenness_distance = [0]

        args = {
            'algorithm': 'dqn',
            'environment': 'simple',
            'agent': 'f1',
            'filename': 'brains/gazebo/f1/config/config_inference_followline_dqn_f1_gazebo.yaml'
        }

        f = open(args['filename'], "r")
        read_file = f.read()

        config_file = yaml.load(read_file, Loader=yaml.FullLoader)

        inference_params = {
            "settings": self.get_settings(config_file),
            "algorithm": self.get_algorithm(config_file, args['algorithm']),
            "inference": self.get_inference(config_file, args['algorithm']),
            "environment": self.get_environment(config_file, args['environment']),
            "agent": self.get_agent(config_file, args['agent']),
        }

        params = InferenceExecutorValidator(**inference_params)

        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        actions = params.environment["actions"]
        env_params["actions"] = actions
        self.environment = LoadEnvVariablesDQNGazebo(config_file)

        self.env = gym.make(self.env_name, **self.environment.environment)

        self.inference_file = params.inference["params"]["inference_file"]
        observation = self.env.reset()
        self.step = 1
        self.state = observation[0]

        self.inferencer = InferencerWrapper("dqn", self.inference_file, env=config_file)

    def get_algorithm(self, config_file: dict, input_algorithm: str) -> dict:
        return {
            "name": input_algorithm,
            "params": config_file["algorithm"][input_algorithm],
        }


    def get_environment(self, config_file: dict, input_env: str) -> dict:
        return {
            "name": input_env,
            "params": config_file["environments"][input_env],
            "actions": config_file["actions"],
        }


    def get_agent(self, config_file: dict, input_agent: str) -> dict:
        return {
            "name": input_agent,
            "params": config_file["agent"][input_agent],
        }


    def get_inference(self, config_file: dict, input_inference: str) -> dict:
        return {
            "name": input_inference,
            "params": config_file["inference"][input_inference],
        }


    def get_settings(self, config_file: dict) -> dict:
        return {
            "name": "settings",
            "params": config_file["settings"],
        }

    def update_frame(self, frame_id, data):
        """Update the information to be shown in one of the GUI's frames.

        Arguments:
            frame_id {str} -- Id of the frame that will represent the data
            data {*} -- Data to be shown in the frame. Depending on the type of frame (rgbimage, laser, pose3d, etc)
        """

        self.handler.update_frame(frame_id, data)


    def execute(self):
        action = np.argmax(self.inferencer.inference(self.state))
        # Execute the action and get feedback
        observation, reward, done, info = self.env.step(action, self.step)
        self.step += 1
        self.state = observation
        image = self.camera.getImage().data

        self.update_frame('frame_0', image)

