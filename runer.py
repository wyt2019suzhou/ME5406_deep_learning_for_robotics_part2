import numpy as np
from abstract import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    def __init__(self, env, model, nsteps, gamma, lam,human_model=None,robot_model=None):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.human_model=human_model
        self.robot_model = robot_model

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            actions, values, neglogpacs = self.model.step(self.obs)#[env,agent]
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            self.obs[:], rewards, self.dones, infos = self.env.step(actions)#infos one stp info
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:#episode info
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)#[step,env,shape]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)#[step,env]
        mb_actions = np.asarray(mb_actions)#[step,env,action_num]
        mb_values = np.asarray(mb_values, dtype=np.float32)#[step,env]
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)#[step,env,agent]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)#[step,env]
        last_values = self.model.value(self.obs)#[env,]

        mb_advs = np.zeros_like(mb_rewards)#[step,env]
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones#[step,env]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]#[env,agent]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values#[step,env]
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),epinfos)

    def coop_run(self):
        mb_obs, mb_rewards, human_mb_actions, human_mb_values, mb_dones, human_mb_neglogpacs,robot_mb_actions, robot_mb_values,robot_mb_neglogpacs = [],[],[],[],[],[],[],[],[]
        epinfos = []
        for _ in range(self.nsteps):
            human_actions, human_values, human_neglogpacs = self.human_model.step(self.obs[:,24:])#[env,agent]
            robot_actions, robot_values, robot_neglogpacs = self.robot_model.step(self.obs[:,:24])  # [env,agent]
            mb_obs.append(self.obs.copy())
            human_mb_actions.append(human_actions)
            human_mb_values.append(human_values)
            human_mb_neglogpacs.append(human_neglogpacs)
            robot_mb_actions.append(robot_actions)
            robot_mb_values.append(robot_values)
            robot_mb_neglogpacs.append(robot_neglogpacs)
            mb_dones.append(self.dones)
            co_action=np.concatenate([robot_actions,human_actions],axis=1)
            self.obs[:], rewards, self.dones, infos = self.env.step(co_action)#infos one stp info
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:#episode info
                    epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)#[step,env,shape]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)#[step,env]
        human_mb_actions = np.asarray(human_mb_actions)#[step,env,action_num]
        human_mb_values = np.asarray(human_mb_values, dtype=np.float32)#[step,env]
        human_mb_neglogpacs = np.asarray(human_mb_neglogpacs, dtype=np.float32)#[step,env,agent]
        robot_mb_actions = np.asarray(robot_mb_actions)  # [step,env,action_num]
        robot_mb_values = np.asarray(robot_mb_values, dtype=np.float32)  # [step,env]
        robot_mb_neglogpacs = np.asarray(robot_mb_neglogpacs, dtype=np.float32)  # [step,env,agent]
        mb_dones = np.asarray(mb_dones, dtype=np.bool)#[step,env]
        human_last_values = self.human_model.value(self.obs[:,24:])#[env,]
        robot_last_values = self.robot_model.value(self.obs[:,:24])  # [env,]

        human_mb_advs = np.zeros_like(mb_rewards)#[step,env]
        robot_mb_advs = np.zeros_like(mb_rewards)
        human_lastgaelam = 0
        robot_lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones#[step,env]
                human_nextvalues = human_last_values
                robot_nextvalues = robot_last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                robot_nextvalues = robot_mb_values[t+1]
                human_nextvalues = human_mb_values[t + 1]
            human_delta = mb_rewards[t] + self.gamma * human_nextvalues * nextnonterminal - human_mb_values[t]#[env,agent]
            robot_delta = mb_rewards[t] + self.gamma * robot_nextvalues * nextnonterminal - robot_mb_values[
                t]  # [env,agent]
            human_mb_advs[t] = human_lastgaelam = human_delta + self.gamma * self.lam * nextnonterminal * human_lastgaelam
            robot_mb_advs[
                t] = robot_lastgaelam = robot_delta + self.gamma * self.lam * nextnonterminal * robot_lastgaelam
        human_mb_returns = human_mb_advs + human_mb_values#[step,env]
        robot_mb_returns = robot_mb_advs + robot_mb_values  # [step,env]
        return (*map(sf01, (mb_obs, human_mb_returns,robot_mb_returns, mb_dones, human_mb_actions, robot_mb_actions,human_mb_values, robot_mb_values,human_mb_neglogpacs,robot_mb_neglogpacs)),epinfos)

def sf01(arr):
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])