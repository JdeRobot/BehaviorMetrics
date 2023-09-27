# This file contains all clasess to parser parameters from config.yaml into training RL


class LoadAlgorithmParams:
    """
    Retrieves Algorithm params
    """

    def __init__(self, config):
        if config["settings"]["algorithm"] == "ddpg":
            self.gamma = config["algorithm"]["ddpg"]["gamma"]
            self.tau = config["algorithm"]["ddpg"]["tau"]
            self.std_dev = config["algorithm"]["ddpg"]["std_dev"]
            self.model_name = config["algorithm"]["ddpg"]["model_name"]
            self.buffer_capacity = config["algorithm"]["ddpg"]["buffer_capacity"]
            self.batch_size = config["algorithm"]["ddpg"]["batch_size"]

        elif config["settings"]["algorithm"] == "dqn":
            self.alpha = config["algorithm"]["dqn"]["alpha"]
            self.gamma = config["algorithm"]["dqn"]["gamma"]
            self.epsilon = config["algorithm"]["dqn"]["epsilon"]
            self.epsilon_discount = config["algorithm"]["dqn"]["epsilon_discount"]
            self.epsilon_min = config["algorithm"]["dqn"]["epsilon_min"]
            self.model_name = config["algorithm"]["dqn"]["model_name"]
            self.replay_memory_size = config["algorithm"]["dqn"]["replay_memory_size"]
            self.min_replay_memory_size = config["algorithm"]["dqn"][
                "min_replay_memory_size"
            ]
            self.minibatch_size = config["algorithm"]["dqn"]["minibatch_size"]
            self.update_target_every = config["algorithm"]["dqn"]["update_target_every"]
            self.memory_fraction = config["algorithm"]["dqn"]["memory_fraction"]
            self.buffer_capacity = config["algorithm"]["dqn"]["buffer_capacity"]
            self.batch_size = config["algorithm"]["dqn"]["batch_size"]

        elif config["settings"]["algorithm"] == "qlearn":
            self.alpha = config["algorithm"]["qlearn"]["alpha"]
            self.gamma = config["algorithm"]["qlearn"]["gamma"]
            self.epsilon = config["algorithm"]["qlearn"]["epsilon"]
            self.epsilon_min = config["algorithm"]["qlearn"]["epsilon_min"]


class LoadEnvParams:
    """
    Retrieves environment parameters: Gazebo, Carla, OpenAI...
    """

    def __init__(self, config):
        if config["settings"]["simulator"] == "gazebo":
            self.env = config["settings"]["env"]
            self.env_name = config["environments"][self.env]["env_name"]
            self.model_state_name = config["environments"][self.env][
                "model_state_name"
            ]
            self.total_episodes = config["settings"]["total_episodes"]
            self.training_time = config["settings"]["training_time"]
            self.save_episodes = config["environments"][self.env][
                "save_episodes"
            ]
            self.save_every_step = config["environments"][self.env][
                "save_every_step"
            ]
            self.estimated_steps = config["environments"][self.env][
                "estimated_steps"
            ]

        elif config["settings"]["simulator"] == "carla":
            pass


class LoadGlobalParams:
    """
    Retrieves Global params from config.yaml
    """

    def __init__(self, config):
        self.stats = {}  # epoch: steps
        self.states_counter = {}
        self.states_reward = {}
        self.ep_rewards = []
        self.actions_rewards = {
            "episode": [],
            "step": [],
            "v": [],
            "w": [],
            "reward": [],
            "center": [],
        }
        self.aggr_ep_rewards = {
            "episode": [],
            "avg": [],
            "max": [],
            "min": [],
            "step": [],
            "epoch_training_time": [],
            "total_training_time": [],
        }
        self.best_current_epoch = {
            "best_epoch": [],
            "highest_reward": [],
            "best_step": [],
            "best_epoch_training_time": [],
            "current_total_training_time": [],
        }
        self.settings = config["settings"]
        self.mode = config["settings"]["mode"]
        self.task = config["settings"]["task"]
        self.algorithm = config["settings"]["algorithm"]
        self.agent = config["settings"]["agent"]
        self.framework = config["settings"]["framework"]
        self.models_dir = f"{config['settings']['models_dir']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}"
        self.logs_tensorboard_dir = f"{config['settings']['logs_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/TensorBoard"
        self.logs_dir = f"{config['settings']['logs_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/logs"
        self.metrics_data_dir = f"{config['settings']['metrics_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/data"
        self.metrics_graphics_dir = f"{config['settings']['metrics_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/graphics"
        self.training_time = config["settings"]["training_time"]
        ####### States
        self.states = config["settings"]["states"]
        self.states_set = config["states"][self.states]
        ####### Actions
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        ####### Rewards
        self.rewards = config["settings"]["rewards"]


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
        self.environment["retrain_dqn_tf_model_name"] = config["retraining"]["dqn"][
            "retrain_dqn_tf_model_name"
        ]

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
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
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
