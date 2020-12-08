from collections import deque#双端队列
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, obs_t, robot_action,human_action, reward, obs_tp1, done):
        if isinstance(done, bool):
            done = 1 if done else 0
        experience = dict(obs_t=obs_t, robot_action=robot_action,human_action=human_action,
                reward=reward, obs_tp1=obs_tp1, done=done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        obs_t = []
        robot_actions = []
        human_actions= []
        rewards = []
        obs_tp1 = []
        done = []
        for experience in experiences:
            obs_t.append(experience['obs_t'])
            robot_actions.append(experience['robot_action'])
            human_actions.append(experience['human_action'])
            rewards.append(experience['reward'])
            obs_tp1.append(experience['obs_tp1'])
            done.append(experience['done'])
        return obs_t, robot_actions,human_actions, rewards, obs_tp1, done