import numpy as np
import tensorflow as tf


VECTOR_OBS_LEN = 24
HIDDEN_LEN_A_1 = 192
HIDDEN_LEN_C_1 = 192
HIDDEN_LEN_A_2 = 96
HIDDEN_LEN_C_2 = 96
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
    def build_P_network(observation,human=False):  # observation=# [batch,agent,action]
        with tf.variable_scope('PI_network', reuse=tf.AUTO_REUSE):
            observation = Net.instance_norm(observation)
            layer1 = tf.layers.dense(observation, HIDDEN_LEN_A_1, name="layer1", activation=tf.nn.tanh,
                                    )
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_A_2, name="layer2", activation=tf.nn.tanh,
                                    )
            if human==True:
                policy = tf.layers.dense(layer2, HUMAN_POLICY_LEN, name="policy",
                                    )
            if human==False:
                policy = tf.layers.dense(layer2, ROBOT_POLICY_LEN, name="policy",
                                         )

        return policy

    @staticmethod
    def build_V_network(observation):#observation=# [batch,agent,action]
        with tf.variable_scope('V_network', reuse=tf.AUTO_REUSE):
            observation=Net.instance_norm(observation)
            layer1 = tf.layers.dense(observation, HIDDEN_LEN_C_1, name="layer1", activation=tf.nn.tanh,
                                     )
            layer2 = tf.layers.dense(layer1, HIDDEN_LEN_C_2, name="layer2", activation=tf.nn.tanh,
                                     )
            value = tf.layers.dense(layer2, 1, name="value")
        return value

    @staticmethod
    def instance_norm(inputs):
        epsilon = 1e-9  # 避免0除数
        mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)
        return tf.div(inputs - mean, tf.sqrt(tf.add(var, epsilon)))

# In[ ]:
