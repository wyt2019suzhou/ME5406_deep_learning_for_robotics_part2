import numpy as np
import tensorflow as tf


VECTOR_OBS_LEN = 24
HIDDEN_LEN_A_1 = 192
HIDDEN_LEN_V_1 = 192
HIDDEN_LEN_Q_1 = 192
HIDDEN_LEN_A_2 = 96
HIDDEN_LEN_V_2 = 96
HIDDEN_LEN_Q_2 = 96
ROBOT_POLICY_LEN=7
HUMAN_POLICY_LEN=4



class Net:
    @staticmethod
    def normalized_columns_initializer(std=1.0):
        def _initializer_one(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer_one

    @staticmethod
    def build_P_network(observation,name):  # observation=# [batch,agent,action]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            observation = Net.instance_norm(observation)
            layer1 = tf.layers.dense(observation, HIDDEN_LEN_A_1, name="layer1", activation=tf.nn.tanh,bias_initializer=tf.constant_initializer(0.0),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   )
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_A_2, name="layer2", activation=tf.nn.tanh,bias_initializer=tf.constant_initializer(0.0),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
            mu = tf.layers.dense(layer2, ROBOT_POLICY_LEN, name="mu", kernel_initializer=tf.contrib.layers.xavier_initializer())
            greedy_action = tf.nn.tanh(mu)
            sigma = tf.layers.dense(
                layer2, ROBOT_POLICY_LEN, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='sigma')
            dist = tf.distributions.Normal(mu, tf.exp(sigma))
            out = tf.reshape(dist.sample(1), [-1, ROBOT_POLICY_LEN])
            out = tf.stop_gradient(out)
            action = tf.nn.tanh(out)
            log_prob = dist.log_prob(out) - tf.log(1 - action ** 2 + 1e-6)
        return action, greedy_action, log_prob

    @staticmethod
    def build_human_P_network(observation,name):  # observation=# [batch,agent,action]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            observation = Net.instance_norm(observation)
            layer1 = tf.layers.dense(observation, HIDDEN_LEN_A_1, name="layer1", activation=tf.nn.tanh,bias_initializer=tf.constant_initializer(0.0),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                   )
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_A_2, name="layer2", activation=tf.nn.tanh,bias_initializer=tf.constant_initializer(0.0),
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
            mu = tf.layers.dense(layer2, HUMAN_POLICY_LEN, name="mu", kernel_initializer=tf.contrib.layers.xavier_initializer())
            greedy_action = tf.nn.tanh(mu)
            sigma = tf.layers.dense(
                layer2, HUMAN_POLICY_LEN, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='sigma')
            dist = tf.distributions.Normal(mu, tf.exp(sigma))
            out = tf.reshape(dist.sample(1), [-1, HUMAN_POLICY_LEN])
            out = tf.stop_gradient(out)
            action = tf.nn.tanh(out)
            log_prob = dist.log_prob(out) - tf.log(1 - action ** 2 + 1e-6)

        return action, greedy_action, log_prob

    @staticmethod
    def build_V_network(observation,name):#observation=# [batch,agent,action]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            observation = Net.instance_norm(observation)
            layer1 = tf.layers.dense(observation, HIDDEN_LEN_V_1, name="layer1", activation=tf.nn.tanh,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.constant_initializer(0.0))
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_V_2, name="layer2", activation=tf.nn.tanh,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.constant_initializer(0.0))
            V = tf.layers.dense(layer2, 1, name="V",kernel_initializer=tf.contrib.layers.xavier_initializer())
        return V

    @staticmethod
    def build_Q_network(observation,action,name):  # observation=# [batch,agent,action]
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            observation = Net.instance_norm(observation)
            input = tf.concat([observation, action], axis=1)
            layer1 = tf.layers.dense(input, HIDDEN_LEN_Q_1, name="layer1", activation=tf.nn.tanh,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.constant_initializer(0.0))
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_Q_2, name="layer2", activation=tf.nn.tanh,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.constant_initializer(0.0))
            Q = tf.layers.dense(layer2, 1, name="Q", kernel_initializer=tf.contrib.layers.xavier_initializer())
        return Q

    @staticmethod
    def instance_norm(inputs):
        epsilon = 1e-9
        mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)
        return tf.div(inputs - mean, tf.sqrt(tf.add(var, epsilon)))

