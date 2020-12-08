import tensorflow as tf
import numpy as np
import gym
import my_envs
from util import set_global_seeds
import os
import time

set_global_seeds(1001)


with tf.Session() as sess:
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_model')
    file_pos=os.path.join(directory, '00350')
    meta_pos = os.path.join(file_pos, 'ppomodel.meta')
    saver = tf.train.import_meta_graph(meta_pos)
    saver.restore(sess, tf.train.latest_checkpoint(file_pos))
    action_op = sess.graph.get_tensor_by_name('ppo_model/add_1:0')
    input = sess.graph.get_tensor_by_name('ppo_model/ob:0')
    num_env=action_op.shape[0]

    env = gym.make('Feeding-v0')
    env.play_show()
    state = env.reset()
    ep_reward=0
    total_step=0
    while True:
        state=np.expand_dims(state, axis=0)
        state=np.repeat(state, num_env, axis=0)
        action = sess.run(action_op, feed_dict={input: state})
        action=action[0,:]
        state2, reward, done, info = env.step(action)
        ep_reward+=reward
        total_step += 1
        state = state2
        if done:
            success=info['success_time']
            fall=info['fall_time']
            print('episode reward: {} | success time: {} | fall time: {} | episode len : {}'.format(ep_reward,success,fall,total_step))
            break

