
import sys
import multiprocessing
import os.path as osp
import os
import tensorflow as tf
import pprint as pp
from arguments import common_arg_parser
from util import get_session,save_state
from learner import learn
from wrap_env import make_vec_env

def train(args):
    total_timesteps = int(args.num_timesteps)
    seed = args.seed
    nsteps=int(args.nsteps)
    ent_coef=args.ent_coef
    vf_coef=args.vf_coef
    p_coef=args.p_coef
    lr=args.lr
    max_grad_norm=args.max_grad_norm
    gamma=args.gamma
    lam=args.lam
    nminibatches=int(args.nminibatches)
    noptepochs=int(args.noptepochs)
    cliprange=args.cliprange
    save_interval=int(args.save_interval)
    env = build_env(args)
    model = learn(env=env, total_timesteps=total_timesteps, seed = seed, nsteps = nsteps, ent_coef = ent_coef, lr = lr,
    vf_coef = vf_coef,p_coef=p_coef,max_grad_norm = max_grad_norm , gamma =gamma, lam = lam, nminibatches = nminibatches, noptepochs = noptepochs, cliprange = cliprange,
    save_interval = save_interval)

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    config = tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=0,inter_op_parallelism_threads=0)#1 or more?
    config.gpu_options.allow_growth = True
    get_session(config=config)

    env = make_vec_env(nenv, args.seed)

    return env

def main():
    # get argument
    tf.reset_default_graph()
    arg_parser = common_arg_parser()
    args= arg_parser.parse_args()
    pp.pprint(vars(args))

    model, env = train(args)
    savepath = osp.join("my_model/", 'final')
    os.makedirs(savepath, exist_ok=True)
    savepath = osp.join(savepath, 'ppomodel')
    save_state(savepath)

    env.close()

    return model

if __name__ == '__main__':
    main()#get argument from terminal
