import tensorflow as tf
import numpy as np
import gym
import my_envs
from util import set_global_seeds
import os

set_global_seeds(1001)


with tf.Session() as sess:
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'my_model_cop')
    file_pos=os.path.join(directory, '00200')
    meta_pos = os.path.join(file_pos, 'ppomodel.meta')
    saver = tf.train.import_meta_graph(meta_pos)
    saver.restore(sess, tf.train.latest_checkpoint(file_pos))
    human_action_op = sess.graph.get_tensor_by_name('human_ppo_model/add_1:0')
    robot_action_op = sess.graph.get_tensor_by_name('robot_ppo_model/add_1:0')
    human_input = sess.graph.get_tensor_by_name('human_ppo_model/ob:0')
    robot_input = sess.graph.get_tensor_by_name('robot_ppo_model/ob:0')
    num_env=human_action_op.shape[0]

    env = gym.make('FeedingCooperation-v0')
    env.play_show()
    state = env.reset()
    ep_reward=0
    total_step=0
    while True:
        state=np.expand_dims(state, axis=0)
        state=np.repeat(state, num_env, axis=0)
        human_action = sess.run(human_action_op, feed_dict={human_input: state[:,24:]})
        robot_action = sess.run(robot_action_op, feed_dict={robot_input: state[:,:24]})
        human_action=human_action[0,:]
        robot_action =robot_action[0, :]
        cop_action=np.concatenate([robot_action,human_action],axis=0)
        state2, reward, done, info = env.step(cop_action)
        ep_reward+=reward
        total_step += 1
        state = state2
        if done:
            success=info['success_time']
            fall=info['fall_time']
            print('episode reward: {} | success time: {} | fall time: {} | episode len : {}'.format(ep_reward,success,fall,total_step))
            break

