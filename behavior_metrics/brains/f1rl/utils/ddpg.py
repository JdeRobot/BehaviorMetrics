#!/usr/bin/env python
import pickle

import numpy as np
import torch
import torch.optim as optim

from brains.f1rl.utils.ddpg_utils.ddpg_step import ddpg_step
from brains.f1rl.utils.ddpg_utils.Policy_ddpg import Policy
from brains.f1rl.utils.ddpg_utils.Value_ddpg import Value
from brains.f1rl.utils.ddpg_utils.ddpg_utils import Memory, get_env_info, check_path, device, FLOAT, ZFilter


class DDPG:
    def __init__(self,
                 env_id,
                 num_states,
                 num_actions,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_v=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 explore_size=10000,
                 step_per_iter=3000,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 action_noise=0.1,
                 seed=1,
                 model_path=None
                 ):
        self.env_id = env_id
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.polyak = polyak
        self.memory = Memory(memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.action_noise = action_noise
        self.model_path = model_path
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""

        self.action_low = -1
        self.action_high = 1 

        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.policy_net = Policy(
            self.num_actions, self.action_high).to(device)
        self.policy_net_target = Policy(
            self.num_actions, self.action_high).to(device)

        self.value_net = Value(64, self.num_actions).to(device)
        self.value_net_target = Value(64, self.num_actions).to(device)

        self.running_state = ZFilter((self.num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_ddpg.p".format(self.env_id))
            self.policy_net, self.value_net, self.running_state = pickle.load(
                open('{}/{}_ddpg.p'.format(self.model_path, self.env_id), "rb"))

        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.value_net_target.load_state_dict(self.value_net.state_dict())

        self.optimizer_p = optim.Adam(
            self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v = optim.Adam(
            self.value_net.parameters(), lr=self.lr_v)

    def choose_action(self, state, noise_scale):
        """select action"""
        state = np.random.rand(1,32,32)
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
        action = action.cpu().numpy()[0]
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action += noise
        action = np.clip(action, -self.action_high, self.action_high)
        return action 

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        state = np.random.rand(1,32,32)
        test_reward = 0
        while True:
            if render:
                self.env.render()
            # state = self.running_state(state)
            action = self.choose_action(state, 0)
            state, reward, done, _ = self.env.step(action)
            state = np.random.rand(1,32,32)
            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def update(self, batch):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by DDPG
        alg_step_stats = ddpg_step(self.policy_net, self.policy_net_target, self.value_net, self.value_net_target, self.optimizer_p,
                                   self.optimizer_v, batch_state, batch_action, batch_reward, batch_next_state, batch_mask,
                                   self.gamma, self.polyak)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.value_net, self.running_state),
                    open('{}/{}_ddpg.p'.format(save_path, self.env_id), 'wb'))
