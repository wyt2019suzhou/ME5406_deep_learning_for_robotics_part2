import argparse


def common_arg_parser():
    """
    Create an argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # agent parameters
    parser.add_argument('--lr', help='network learning rate', default=  lambda f: 3.0e-4 * f)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--lam', help='discount factor for advantage', default=0.95)
    parser.add_argument('--cliprange', help='clip range for ratio', default=0.2)
    parser.add_argument('--max_grad_norm', help='clip gradient', default=0.5)
    parser.add_argument('--ent_coef', help='entropy discount factor', default=0.001)
    parser.add_argument('--vf_coef', help='value loss discount factor', default=0.5)
    parser.add_argument('--p_coef', help='policy loss discount factor', default=1000)

    # run parameters
    parser.add_argument('--nsteps', help='each environment act step', default=500)
    parser.add_argument('--nminibatches', help='number of minibatch in one train batch', default=5)
    parser.add_argument('--noptepochs', help='number of reuse experience ', default=10)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel', default=5, type=int)
    parser.add_argument('--num_timesteps', help='max number of total step',type=float, default=1e7)

    #other parameters
    parser.add_argument('--seed', help='random seed', type=int, default=1001)
    parser.add_argument('--save_interval', help='save model interval', type=int, default=50)

    return parser


