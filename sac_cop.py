
import argparse#解析命令行参数
import pprint as pp#美观打印
import datetime
import numpy as np
import tensorflow as tf
from net_sac import Net
import gym
import my_envs
from buffer_cop import ReplayBuffer
import os
import time
from tqdm import tqdm
import os.path as osp
from util import set_global_seeds,save_state



OBS_LEN = 45
ROBOT_OBS_LEN=24
HUMAN_OBS_LEN=21
ROBOT_ACTION = 7
HUMAN_ACTION = 4



class Robot_Actor_Critic(object):
    def __init__(self, sess,actor_learning_rate,critic_learning_rate,value_learning_rate, reg_factor,gamma, tau,value_weight,critic_weight,actor_weight,all_rl,max_step,mini_batch_num):
        self.sess = sess
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.value_learning_rate = value_learning_rate
        self.tau = tau
        self.reg_factor=reg_factor
        self.gamma=gamma
        self.value_weight=value_weight
        self.critic_weight=critic_weight
        self.actor_weight=actor_weight
        self.all_rl=all_rl
        self.max_update=max_step-100*mini_batch_num

        with tf.variable_scope('robot', reuse=tf.AUTO_REUSE):
            self.actor_input,self.action, self.greedy_action, self.log_prob= self.create_actor_network("actor")
            self.actor_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'robot/actor')


            self.critic_input,self.input_action, self.q= self.create_critic_network("critic")
            self.critic_actor_input,  self.q_a = self.create_critic_actor_network("critic")
            self.critic_network_params =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'robot/critic')

            self.value_input,self.v = self.create_value_network("value")
            self.value_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'robot/value')

            self.target_inputs, self.target_v = self.create_value_network("taret_value")
            self.target_value_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'robot/taret_value')

            self.all_parameter=[]
            self.all_parameter.extend(self.actor_network_params)
            self.all_parameter.extend(self.critic_network_params)
            self.all_parameter.extend(self.value_network_params)

            with tf.variable_scope('update_value_target'):
                self.update_value_target = [
                    self.target_value_network_params[i].assign(tf.multiply(self.value_network_params[i], self.tau)
                                                                 + tf.multiply(self.target_value_network_params[i],
                                                                               1. - self.tau))
                    for i in range(len(self.target_value_network_params))]


            self.rew=tf.placeholder(tf.float32, [None], name='reward')
            self.done_mask= tf.placeholder(tf.float32, [None], name='done')


            with tf.name_scope('value_loss'):
                value_target = self.q- self.log_prob
                self.value_loss = tf.reduce_mean( 0.5 * tf.square(self.v - tf.stop_gradient(value_target)))

            with tf.name_scope('critic_loss'):
                critic_target = self.rew + self.gamma * self.target_v * (1.0 - self.done_mask)#Q
                self.critic_loss = tf.reduce_mean(0.5 * tf.square(self.q - tf.stop_gradient(critic_target)))

            with tf.name_scope('policy_loss'):
                policy_target = self.q_a- self.v
                actor_loss = 0.5 * tf.reduce_mean(
                    self.log_prob * tf.stop_gradient(self.log_prob - policy_target))#log_pi_t entropy
                self.actor_loss = actor_loss

            self.all_loss=self.value_loss*self.value_weight+self.critic_loss*self.critic_weight +self.actor_loss*self.actor_weight
            current_step=tf.Variable(0,trainable=False)
            learning_rate=tf.train.exponential_decay(self.all_rl,current_step,self.max_update,0.96)
            self.all_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.all_train_op = self.all_optimizer.minimize(self.all_loss, var_list=self.all_parameter,global_step=current_step)#can not use place hold as rl

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None,ROBOT_OBS_LEN), name="actor_inputs")
        action, greedy_action, log_prob= Net.build_P_network(inputs,name)
        return inputs,action, greedy_action, log_prob

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None,  ROBOT_OBS_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None,ROBOT_ACTION), name="critic_action")
        q= Net.build_Q_network( inputs, action,name)
        return inputs, action,q

    def create_critic_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, ROBOT_OBS_LEN), name="critic_actor_inputs")
        q= Net.build_Q_network( inputs,self.action,name)
        return inputs,q

    def create_value_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, ROBOT_OBS_LEN), name="value_inputs")
        v = Net.build_V_network(inputs, name)
        return inputs, v

    def actor_predict(self, inputs):
        return self.sess.run([self.action, self.greedy_action], feed_dict={
            self.actor_input: inputs
        })
    def all_train(self,observation,observation_n,acton,reward,done):
        return self.sess.run([self.actor_loss,self.critic_loss,self.value_loss,self.all_loss,self.all_train_op],feed_dict={
            self.rew:reward,
            self.done_mask:done,
        self.target_inputs:observation_n,
        self.critic_input:observation,
        self.actor_input: observation,
        self.critic_actor_input: observation,
        self.value_input: observation,
        self.input_action:acton,
        })

    def update_target_network(self):
        self.sess.run(self.update_value_target)

class Human_Actor_Critic(object):
    def __init__(self, sess,actor_learning_rate,critic_learning_rate,value_learning_rate, reg_factor,gamma, tau,value_weight,critic_weight,actor_weight,all_rl,max_step,mini_batch_num):
        self.sess = sess
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.value_learning_rate = value_learning_rate
        self.tau = tau
        self.reg_factor=reg_factor
        self.gamma=gamma
        self.value_weight=value_weight
        self.critic_weight=critic_weight
        self.actor_weight=actor_weight
        self.all_rl=all_rl
        self.max_update=max_step-100*mini_batch_num

        with tf.variable_scope('human', reuse=tf.AUTO_REUSE):
            self.actor_input,self.action, self.greedy_action, self.log_prob= self.create_actor_network("actor")
            self.actor_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'human/actor')


            self.critic_input,self.input_action, self.q= self.create_critic_network("critic")
            self.critic_actor_input,  self.q_a = self.create_critic_actor_network("critic")
            self.critic_network_params =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'human/critic')

            self.value_input,self.v = self.create_value_network("value")
            self.value_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'human/value')

            self.target_inputs, self.target_v = self.create_value_network("taret_value")
            self.target_value_network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'human/taret_value')

            self.all_parameter=[]
            self.all_parameter.extend(self.actor_network_params)
            self.all_parameter.extend(self.critic_network_params)
            self.all_parameter.extend(self.value_network_params)

            with tf.variable_scope('update_value_target'):
                self.update_value_target = [
                    self.target_value_network_params[i].assign(tf.multiply(self.value_network_params[i], self.tau)
                                                                 + tf.multiply(self.target_value_network_params[i],
                                                                               1. - self.tau))
                    for i in range(len(self.target_value_network_params))]


            self.rew=tf.placeholder(tf.float32, [None], name='reward')
            self.done_mask= tf.placeholder(tf.float32, [None], name='done')


            with tf.name_scope('value_loss'):
                value_target = self.q- self.log_prob
                self.value_loss = tf.reduce_mean( 0.5 * tf.square(self.v - tf.stop_gradient(value_target)))

            with tf.name_scope('critic_loss'):
                critic_target = self.rew + self.gamma * self.target_v * (1.0 - self.done_mask)#Q
                self.critic_loss = tf.reduce_mean(0.5 * tf.square(self.q - tf.stop_gradient(critic_target)))

            with tf.name_scope('policy_loss'):
                policy_target = self.q_a- self.v
                actor_loss = 0.5 * tf.reduce_mean(
                    self.log_prob * tf.stop_gradient(self.log_prob - policy_target))#log_pi_t entropy
                self.actor_loss = actor_loss

            self.all_loss=self.value_loss*self.value_weight+self.critic_loss*self.critic_weight +self.actor_loss*self.actor_weight
            current_step=tf.Variable(0,trainable=False)
            learning_rate=tf.train.exponential_decay(self.all_rl,current_step,self.max_update,0.96)
            self.all_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.all_train_op = self.all_optimizer.minimize(self.all_loss, var_list=self.all_parameter,global_step=current_step)#can not use place hold as rl

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None,HUMAN_OBS_LEN), name="actor_inputs")
        action, greedy_action, log_prob= Net.build_human_P_network(inputs,name)
        return inputs,action, greedy_action, log_prob

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None,  HUMAN_OBS_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None,HUMAN_ACTION), name="critic_action")
        q= Net.build_Q_network( inputs, action,name)
        return inputs, action,q

    def create_critic_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, HUMAN_OBS_LEN), name="critic_actor_inputs")
        q= Net.build_Q_network( inputs,self.action,name)
        return inputs,q

    def create_value_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, HUMAN_OBS_LEN), name="value_inputs")
        v = Net.build_V_network(inputs, name)
        return inputs, v

    def actor_predict(self, inputs):
        return self.sess.run([self.action, self.greedy_action], feed_dict={
            self.actor_input: inputs
        })
    def all_train(self,observation,observation_n,acton,reward,done):
        return self.sess.run([self.actor_loss,self.critic_loss,self.value_loss,self.all_loss,self.all_train_op],feed_dict={
            self.rew:reward,
            self.done_mask:done,
        self.target_inputs:observation_n,
        self.critic_input:observation,
        self.actor_input: observation,
        self.critic_actor_input: observation,
        self.value_input: observation,
        self.input_action:acton,
        })

    def update_target_network(self):
        self.sess.run(self.update_value_target)

def train(sess, env, args, robot_actor_critic,human_actor_critic):
    sess.run(tf.global_variables_initializer())
    global_summary = tf.summary.FileWriter('summaries/' + 'feeding_sac_all' + datetime.datetime.now().strftime('%d-%m-%y%H%M'),sess.graph)
    robot_actor_critic.update_target_network()
    human_actor_critic.update_target_network()


    replay_buffer = ReplayBuffer(int(args['buffer_size']))
    pbar = tqdm(total=int(args['max_steps']), dynamic_ncols=True)
    tfirststart=time.perf_counter()
    total_step=0

    while total_step<int(args['max_steps']):
        state=env.reset()
        episode_reward=0
        end_step=0
        while True:
            robot_action, robot_greedy_action= robot_actor_critic.actor_predict([state[:24]])
            human_action, human_greedy_action = human_actor_critic.actor_predict([state[24:]])
            robot_action=robot_action[0]
            robot_greedy_action=robot_greedy_action[0]
            human_action = human_action[0]
            human_greedy_action = human_greedy_action[0]
            cop_action = np.concatenate([robot_action, human_action], axis=0)
            state2, reward, done, info = env.step(cop_action)
            episode_reward+=reward
            end_step += 1
            total_step+=1

            replay_buffer.add(state, robot_action,human_action, reward, state2, done)


            state = state2

            if total_step > 100*int(args['minibatch_size']):
                batch_state, batch_robot_actions, batch_human_actions,batch_rewards, batch_state2, batch_dones =replay_buffer.sample(int(args['minibatch_size']))
                batch_state=np.array(batch_state)
                batch_state2 = np.array(batch_state2)
                robot_actor_loss,robot_critic_loss,robot_value_loss,robot_all_loss,_=robot_actor_critic.all_train(batch_state[:,:24],batch_state2[:,:24],batch_robot_actions,batch_rewards,batch_dones)
                robot_actor_critic.update_target_network()
                human_actor_loss, human_critic_loss, human_value_loss, human_all_loss, _ = human_actor_critic.all_train(batch_state[:,24:], batch_state2[:,24:],
                                                                                          batch_human_actions, batch_rewards,
                                                                                          batch_dones)
                human_actor_critic.update_target_network()

                summary = tf.Summary()
                summary.value.add(tag='robot_loss/value_loss', simple_value=robot_value_loss)
                summary.value.add(tag='robot_loss/critic_loss', simple_value=robot_critic_loss)
                summary.value.add(tag='robot_loss/actor_loss', simple_value= robot_actor_loss)
                summary.value.add(tag='robot_loss/total_loss', simple_value=robot_all_loss)
                summary.value.add(tag='human_loss/value_loss', simple_value=human_value_loss)
                summary.value.add(tag='human_loss/critic_loss', simple_value=human_critic_loss)
                summary.value.add(tag='human_loss/actor_loss', simple_value=human_actor_loss)
                summary.value.add(tag='human_loss/total_loss', simple_value=human_all_loss)
                global_summary.add_summary(summary,total_step)
                global_summary.flush()

            if total_step%1000000==0 and total_step!=0:
                tnow = time.perf_counter()
                print('consume time', tnow - tfirststart)
                savepath = osp.join("my_model_sac_cop/", '%.5i' %total_step )
                os.makedirs(savepath, exist_ok=True)
                savepath = osp.join(savepath, 'sacmodel')
                print('Saving to', savepath)
                save_state(savepath)

            if done:
                success_time=env.success_time()
                fall_time=env.fall_times()
                msg = 'step: {},episode reward: {},episode len: {},success_time: {},fall_time: {}'
                pbar.update(total_step)
                pbar.set_description(msg.format(total_step, episode_reward, end_step,success_time,fall_time))
                summary = tf.Summary()
                summary.value.add(tag='Perf/Reward', simple_value= episode_reward)
                summary.value.add(tag='Perf/episode_len', simple_value=end_step)
                summary.value.add(tag='Perf/success_time', simple_value=success_time)
                summary.value.add(tag='Perf/fall_time', simple_value=fall_time)
                global_summary.add_summary(summary, total_step)
                global_summary.flush()
                break



def main():

    args = parse_arg()
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        env = gym.make('FeedingCooperation-v0')
        set_global_seeds(int(args['random_seed']))
        robot_actor_critic_entropy = Robot_Actor_Critic(sess,float(args['actor_lr']),float(args['critic_lr']),float(args['value_lr']), float(args['reg_factor']),float(args['gamma']), float(args['tau']),
                                            float(args['value_weight']),float(args['critic_weight']),float(args['actor_weight']),float(args['all_lr']),float(args['max_steps']),float(args['minibatch_size']))
        human_actor_critic_entropy = Human_Actor_Critic(sess, float(args['actor_lr']), float(args['critic_lr']),
                                                        float(args['value_lr']), float(args['reg_factor']),
                                                        float(args['gamma']), float(args['tau']),
                                                        float(args['value_weight']), float(args['critic_weight']),
                                                        float(args['actor_weight']), float(args['all_lr']),
                                                        float(args['max_steps']), float(args['minibatch_size']))
        train(sess, env, args, robot_actor_critic_entropy,human_actor_critic_entropy)
        savepath = osp.join("my_model_sac_cop/", 'final')
        os.makedirs(savepath, exist_ok=True)
        savepath = osp.join(savepath, 'sacmodel')
        save_state(savepath)


def parse_arg():
    parser = argparse.ArgumentParser(description='provide arguments for sac agent')#解析对象

    parser.add_argument('--actor_lr', help='actor network learning rate', default=3 * 1e-4)
    parser.add_argument('--critic_lr', help='critic network learning rate', default=3 * 1e-4)
    parser.add_argument('--value_lr', help='value network learning rate', default=3 * 1e-4)
    parser.add_argument('--all_lr', help='network learning rate', default= 3 * 1e-4)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=1e-2)
    parser.add_argument('--reg_factor', help='regular parameter', default=1e-3)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=10 ** 7)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=128)
    parser.add_argument('--value_weight', help='weight of vaule loss', default=1.0)
    parser.add_argument('--critic_weight', help='weight of critic loss', default=0.1)
    parser.add_argument('--actor_weight', help='weight of actor loss', default=1.0)

    # run parameters
    parser.add_argument('--random_seed', help='random seed for repeatability', default=1001)
    parser.add_argument('--max_steps', help='max num of episodes to do while training', default=10 ** 7)

    args = vars(parser.parse_args())
    pp.pprint(args)
    return args



if __name__ == '__main__':
    main()

