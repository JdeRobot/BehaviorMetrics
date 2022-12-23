import pickle
import random

import numpy as np


class QLearn:
    def __init__(self, actions, epsilon=0.99, alpha=0.8, gamma=0.9):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def getQValues(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        """
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        """
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def selectAction(self, state, return_q=False):
        q = [self.getQValues(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [
                q[i] + random.random() * mag - 0.5 * mag
                for i in range(len(self.actions))
            ]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQValues(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma * maxqnew)

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def inference(self, state, return_q=False):
        q = [self.getQValues(state, a) for a in self.actions]
        maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        return action

    def load_model(self, file_path, actions_path):

        qlearn_file = open(file_path, "rb")
        actions_file = open(actions_path, "rb")

        self.q = pickle.load(qlearn_file)
        # TODO it may be possible to infer the actions from the model. I don know enough to assume that for every algorithm
        self.actions = pickle.load(actions_file)

        print(f"\n\nMODEL LOADED.")
        print(f"    - Loading:    {file_path}")
        print(f"    - Model size: {len(self.q)}")

    def updateEpsilon(self, epsilon):
        self.epsilon = epsilon
        return self.epsilon