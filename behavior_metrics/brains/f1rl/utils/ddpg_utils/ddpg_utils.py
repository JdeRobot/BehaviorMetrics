#!/usr/bin/env python

import random
from collections import namedtuple, deque
import gym
from gym.spaces import Discrete
import os
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLOAT = torch.FloatTensor
DOUBLE = torch.DoubleTensor
LONG = torch.LongTensor


def to_device(*args):
    return [arg.to(device) for arg in args]


def get_flat_params(model: nn.Module):
    """
    get tensor flatted parameters from model
    :param model:
    :return: tensor
    """
    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):
    """
    get flatted grad of parameters from the model
    :param model:
    :return: tensor
    """
    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])


def set_flat_params(model, flat_params):
    """
    set tensor flatted parameters to model
    :param model:
    :param flat_params: tensor
    :return:
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh


Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob'))


def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")

def get_env_space(env_id):
    env = gym.make(env_id)
    num_states = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    return env, num_states, num_actions


def get_env_info(env_id, unwrap=False):
    env = gym.make(env_id)
    if unwrap:
        env = env.unwrapped
    num_states = env.observation_space.shape[0]
    env_continuous = False
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]
        env_continuous = True

    return env, env_continuous, num_states, num_actions


class Memory(object):
    def __init__(self, size=None):
        self.memory = deque(maxlen=size)

    # save item
    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory.clear()

    def append(self, other):
        self.memory += other.memory

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:  # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)

# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
