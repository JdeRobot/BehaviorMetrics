#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import yaml
import gym

from gym.envs.registration import register
from brains.gazebo.f1.rl_utils.inference import InferencerWrapper


# F1 envs
if 'F1Env-v0' not in gym.envs.registry.env_specs:
    gym.envs.register(
        id='F1Env-v0',
        entry_point='brains.gazebo.f1.rl_utils.models:F1Env',
        # More arguments here
    )
else:
    print("Environment F1Env-v0 is already registered.")


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
            'algorithm': 'qlearn',
            'environment': 'simple', 
            'agent': 'f1',
            'filename': 'brains/gazebo/f1/config/config_f1_qlearn.yaml'
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

        self.env = gym.make(self.env_name, **env_params)

        outdir = "./logs/f1_qlearn_gym_experiments/"
        self.env = gym.wrappers.Monitor(self.env, outdir, force=True)
        observation = self.env.reset()
        self.state = "".join(map(str, observation))

        self.inference_file = params.inference["params"]["inference_file"]
        self.actions_file = params.inference["params"]["actions_file"]

        self.inferencer = InferencerWrapper("qlearn", self.inference_file, self.actions_file)



    def get_algorithm(self, config_file: dict, input_algorithm: str) -> dict:
        return {
            "name": input_algorithm,
            "params": config_file["algorithm"][input_algorithm],
        }


    def get_environment(self, config_file: dict, input_env: str) -> dict:
        return {
            "name": input_env,
            "params": config_file["environments"][input_env],
            "actions": config_file["actions"]
            .get("available_actions", None)
            .get(config_file["actions"].get("actions_set", None), None),
            "actions_set": config_file["actions"].get("actions_set", None),
            "actions_number": config_file["actions"].get("actions_number", None),
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
        action = self.inferencer.inference(self.state)
        # Execute the action and get feedback
        observation, reward, done, info = self.env.step(action)

        self.state = "".join(map(str, observation))

        image = self.camera.getImage().data
        
        self.update_frame('frame_0', image)
        
