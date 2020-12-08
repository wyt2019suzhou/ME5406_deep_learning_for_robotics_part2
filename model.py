import tensorflow as tf
from policies import PolicyWithValue as policy
from util import get_session


class Model(object):

    def __init__(self, env, nbatch_act, nbatch_train,ent_coef, vf_coef,p_coef, max_grad_norm,human=False,robot=False):
        self.sess = sess = get_session()

        if human:
            with tf.variable_scope('human_ppo_model', reuse=tf.AUTO_REUSE):

                act_model = policy(env,nbatch_act, sess,human=True,robot=False)

                train_model = policy(env,nbatch_train, sess,human=True,robot=False)
        if robot:
            with tf.variable_scope('robot_ppo_model', reuse=tf.AUTO_REUSE):

                act_model = policy(env, nbatch_act, sess,human=False,robot=True)

                train_model = policy(env, nbatch_train, sess,human=False,robot=True)
        if human==False and robot==False:
            with tf.variable_scope('ppo_model', reuse=tf.AUTO_REUSE):
                act_model = policy(env, nbatch_act, sess)

                train_model = policy(env, nbatch_train, sess)
        if human:
            self.A = A = tf.placeholder(tf.float32, shape=(nbatch_train,4))
        if robot:
            self.A = A = tf.placeholder(tf.float32, shape=(nbatch_train,7))
        if human==False and robot==False:
            self.A = A =tf.placeholder(tf.float32, shape=(nbatch_train,) + env.action_space.shape) #action[batch,action]
        self.ADV = ADV = tf.placeholder(tf.float32,shape=[None])#[batch]
        self.R = R = tf.placeholder(tf.float32,shape=[None])#[batch]

        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32,[None])  # [batch,action]

        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32,[None])#[None]
        self.LR = LR = tf.placeholder(tf.float32, [])
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)#come from comparsion between onehot A and train network output

        entropy = tf.reduce_mean(train_model.pd.entropy())#[]

        vpred = train_model.vf#[batch,]
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)#[batch,]
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)#[batch,]
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)#[batch,]

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))#[]

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)#[batch,agent]
        pg_losses = -ADV * ratio#[batch,action]#######[500] uncompile with [500,7]

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)#[batch,action]

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))#[]
        approxkl = .5 * tf.reduce_mean(tf.square( neglogpac - OLDNEGLOGPAC))#[]
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))#[]choose clipprange

        # Total loss
        loss = pg_loss*p_coef - entropy * ent_coef + vf_loss * vf_coef#[]

        if human:
            params = tf.trainable_variables('human_ppo_model')
        if robot:
            params = tf.trainable_variables('robot_ppo_model')
        if human==False and robot==False:
            params = tf.trainable_variables('ppo_model')
        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        grads_and_var = self.trainer.compute_gradients(loss, params)#none
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)#grad-norm,the norm of grad befor clip,None
        grads_and_var = list(zip(grads, var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)#None
        if human:
            self.loss_names = ['human_policy_loss', 'human_value_loss', 'human_policy_entropy', 'human_approxkl', 'human_clipfrac','human_total_loss']
        if robot:
            self.loss_names = ['robot_policy_loss', 'robot_value_loss', 'robot_policy_entropy', 'robot_approxkl', 'robot_clipfrac', 'robot_total_loss']
        if human==False and robot==False:
            self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'total_loss']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac,loss]


        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value



    def train(self, lr, cliprange, obs, returns,  actions, values, neglogpacs):
        advs = returns - values#[batch]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)#[batch,]

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
