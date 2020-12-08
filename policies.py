import tensorflow as tf
import distributions
from net import Net



class PolicyWithValue(object):

    def __init__(self, env, nbatch, sess=None,human=False,robot=False):
        ob_space = env.observation_space
        a_space=env.action_space
        if human:
            self.X = tf.placeholder(shape=(nbatch,21), dtype=tf.float32, name='ob')#[BATCH,OBS]
        if robot:
            self.X = tf.placeholder(shape=(nbatch,24), dtype=tf.float32, name='ob')  # [BATCH,OBS]
        if human==False and robot==False:
            self.X = tf.placeholder(shape=(nbatch,) + ob_space.shape, dtype=tf.float32, name='ob')  # [BATCH,OBS]
        self.vf=Net.build_V_network(self.X)

        if human:
            latent= Net.build_P_network(self.X,human=True)#[batch,num,1],#[batch,num,5]
        if human==False:
            latent = Net.build_P_network(self.X)
        if human:
            self.pdtype =distributions.DiagGaussianPdType(4)
        if robot:
            self.pdtype = distributions.DiagGaussianPdType(7)
        if human == False and robot == False:
            self.pdtype = distributions.DiagGaussianPdType(a_space.shape[0])
        self.pd = self.pdtype.pdfromlatent(latent)  # pi[batch,action]

        self.action = self.pd.sample()  # one number[batch,]
        self.neglogp = self.pd.neglogp(self.action)
        self.vf=tf.squeeze(self.vf)#[batch,]
        self.sess = sess

    def _evaluate(self, variables, observation):
        sess = self.sess
        feed_dict = {self.X: observation}
        return sess.run(variables, feed_dict)

    def step(self, observation):
        a, v, neglogp = self._evaluate([self.action, self.vf, self.neglogp], observation)
        return a, v, neglogp

    def value(self, ob):
        return self._evaluate(self.vf, ob)





