import numpy as np
import tensorflow as tf
from brains.f1rl.utils.ddpg_utils.actor import Actor
from brains.f1rl.utils.ddpg_utils.critic import Critic
from brains.f1rl.utils.ddpg_utils.utils import OrnsteinUhlenbeckActionNoise, ReplayBuffer
from copy import deepcopy

class DDPGAgent:

    def __init__(self,
                sess = None,
                state_dim = None,
                action_dim = None,
                training_batch = None,
                epsilon_decay = None,
                gamma = 0.99,
                tau = 0.001):

        self._sess = sess
        self._state_dim = state_dim
        self._action_dim = action_dim

        self._actor = Actor(sess=self._sess,
                            state_dim=self._state_dim,
                            action_dim=self._action_dim,
                            hidden_sizes=[32, 16, 16],
                            output_activation=tf.nn.tanh,
                            scope = 'actor')

        self._critic = Critic(sess=sess,
                            state_dim=self._state_dim,
                            action_dim=self._action_dim,
                            latent_dim=2*self._action_dim,
                            hidden_sizes=[32, 16, 32, 32],
                            output_activation=tf.nn.relu,
                            scope = 'critic')

        self._target_actor = Actor(sess=self._sess,
                            state_dim=self._state_dim,
                            action_dim=self._action_dim,
                            hidden_sizes=[32, 16, 16],
                            output_activation=tf.nn.tanh,
                            scope = 'target_actor')

        self._target_critic = Critic(sess=sess,
                            state_dim=self._state_dim,
                            action_dim=self._action_dim,
                            latent_dim=2*self._action_dim,
                            hidden_sizes=[32, 16, 32, 32],
                            output_activation=tf.nn.relu,
                            scope = 'target_critic')

        self._actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self._action_dim))

        self._memory_limit = 20000
        self._memory_batch = training_batch
        self._memory = ReplayBuffer(buffer_size=self._memory_limit)

        self._epsilon = 1.0
        self._epsilon_decay = epsilon_decay

        self._discount_factor = gamma
        self._tau = tau

        self._hard_update_actor = self.update_target_network(self._actor.network_params, self._target_actor.network_params, tau=1.0)
        self._hard_update_critic = self.update_target_network(self._actor.network_params, self._target_actor.network_params, tau=1.0)

        self._soft_update_actor = self.update_target_network(self._actor.network_params, self._target_actor.network_params, tau=self._tau)
        self._soft_update_critic = self.update_target_network(self._actor.network_params, self._target_actor.network_params, tau=self._tau)

    def memorize(self, state, action, reward, terminal, next_state):

        self._memory.add(state, action, reward, terminal, next_state)

    def bellman(self, rewards, q_values, terminal, gamma):

        critic_target = np.random.rand(1)

        if terminal:
            critic_target[0] = rewards
        else:
            critic_target[0] = rewards + gamma * q_values
        return critic_target

    def predict(self, state):

        action = self._actor.predict(state=state)[0]

        action += self._actor_noise()

        action = np.clip(action, -1.0, 1.0)

        return action

    def train(self):

        if self._memory.size() > self._memory_batch:

            s_batch, a_batch, r_batch, t_batch, next_s_batch = self._memory.sample_batch(self._memory_batch)

            for i in range(self._memory_batch):

                q_value = self._target_critic.predict(state=next_s_batch[i],
                                                        action=self._target_actor.predict(state=next_s_batch[i]))

                critic_target = self.bellman(r_batch[i], q_value, t_batch[i], self._discount_factor)

                self._critic.update(state=s_batch[i], action=a_batch[i], critic_target=critic_target)

                actions = self._actor.predict(state=s_batch[i])
                grads = self._critic.gradients(state=s_batch[i], action=actions)

                self._actor.update(state=s_batch[i], action_gradients=grads)

            self._sess.run(self._soft_update_actor)
            self._sess.run(self._soft_update_critic)

        else:
            pass

    def update_target_network(self, network_params, target_network_params, tau):     

        op_holder = []
        for from_var,to_var in zip(network_params, target_network_params):
            op_holder.append(to_var.assign((tf.multiply(from_var, tau) + tf.multiply(to_var, 1. - tau))))        
        
        return op_holder

    def _hard_update(self):

        self._sess.run(self._hard_update_actor)
        self._sess.run(self._hard_update_critic)
