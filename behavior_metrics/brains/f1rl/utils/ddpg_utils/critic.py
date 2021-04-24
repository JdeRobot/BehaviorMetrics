import numpy as np
import tensorflow as tf


class Critic:

    def __init__(self,
                sess=None,
                state_dim = None,
                action_dim = None,
                latent_dim = None,
                hidden_sizes = None,
                learning_rate = 0.003,
                hidden_activation = tf.nn.relu,
                output_activation = tf.nn.relu,
                w_init=tf.contrib.layers.xavier_initializer(),
                b_init=tf.zeros_initializer(),
                scope = 'critic'
                ):

        self._sess = sess

        ############################# Critic Layer ###########################

        with tf.variable_scope(scope):
            
            self._state = tf.placeholder(dtype=tf.float32, shape=state_dim, name='state')
            self._action = tf.placeholder(dtype=tf.float32, shape=action_dim, name='action')
            self._return_target = tf.placeholder(dtype=tf.float32, shape=1, name='target')

            with tf.GradientTape() as tape:

                tape.watch([self._state, self._action])

                layer = tf.layers.conv1d(inputs=tf.expand_dims(self._state, 0),
                                        filters=hidden_sizes[0],
                                        kernel_size=4,
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_in')

                for i in range(len(hidden_sizes)-3):

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
                                        units=latent_dim, 
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='layer_out_latent')

                layer = tf.layers.flatten(layer)

                action_layer =  tf.concat([layer, tf.reshape(self._action, (1,action_dim))], 1)

                action_layer = tf.layers.dense(inputs=action_layer, 
                                        units=hidden_sizes[i+2], 
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='action_layer_'+str(i+2))
                
                action_layer = tf.layers.dense(inputs=action_layer, 
                                        units=hidden_sizes[i+3], 
                                        activation=hidden_activation, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='action_layer_'+str(i+3))

                action_layer = tf.layers.dense(inputs=action_layer, 
                                        units=1, 
                                        activation=None, 
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init,
                                        name='action_layer_out')

                self.layer_out = tf.layers.flatten(action_layer)

                self.predicted_qvalue = tf.squeeze(self.layer_out)

                self._loss = tf.reduce_mean(tf.square(self.predicted_qvalue - self._return_target))

                self._action_gradients = tape.gradient(self.predicted_qvalue, self._action)

                self.network_params = tf.trainable_variables()

        self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)

    def gradients(self, state, action):
        return self._sess.run(self._action_gradients, {self._state: state, self._action: action})

    def predict(self, state, action):
        return self._sess.run(self.predicted_qvalue, {self._state: state, self._action: action})

    def update(self, state, action, critic_target):

        self._sess.run(self._train_op, 
                        {self._state: state,
                        self._action: action,
                        self._return_target: critic_target})
