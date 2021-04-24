import numpy as np
import tensorflow as tf


class Actor:

    def __init__(self,
                sess=None,
                state_dim = None,
                action_dim = None,
                hidden_sizes = None,
                learning_rate = 0.0003,
                hidden_activation = tf.nn.relu,
                output_activation = tf.nn.tanh,
                w_init=tf.contrib.layers.xavier_initializer(),
                b_init=tf.zeros_initializer(),
                scope = 'actor'
                ):

        self._sess = sess
        self._state = tf.placeholder(dtype=tf.float32, shape=state_dim, name='state')
        self._action_grads = tf.placeholder(dtype=tf.float32, shape=action_dim, name='action_grads')

        ############################# Actor Layer ###########################

        with tf.variable_scope(scope):

            layer = tf.layers.conv1d(inputs=tf.expand_dims(self._state, 0),
                                    filters=hidden_sizes[0],
                                    kernel_size=4,
                                    activation=hidden_activation, 
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init,
                                    name='layer_in')

            for i in range(len(hidden_sizes)-1):

                layer = tf.layers.conv1d(inputs=layer, 
                                        filters=hidden_sizes[i+1],
                                        kernel_size=3,
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_'+str(i+1))

            layer = tf.layers.flatten(inputs=layer, 
                                    name='layer_flatten')

            layer = tf.layers.dense(inputs=layer, 
                                    units=action_dim, 
                                    activation=output_activation, 
                                    kernel_initializer=w_init,
                                    bias_initializer=b_init,
                                    name='layer_out')

            self.layer_out = tf.layers.flatten(layer)

            self.predicted_action = tf.squeeze(self.layer_out)

            self.network_params = tf.trainable_variables()

        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        params_grad = tf.gradients(self.predicted_action, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), -self._action_grads)
        grads = zip(params_grad, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        self._train_op = self._optimizer.apply_gradients(grads)

    def predict(self, state):
        return self._sess.run(self.predicted_action, {self._state: state})

    def update(self, state, action_gradients):

        self._sess.run(self._train_op, 
                        {self._state: state,
                        self._action_grads: action_gradients})
