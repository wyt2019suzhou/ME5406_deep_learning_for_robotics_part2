import os
import time
import numpy as np
import os.path as osp
from collections import deque
from util import explained_variance, set_global_seeds,get_session,save_state,initialize
from runer import Runner
from model import Model
import tensorflow as tf
import datetime
from tqdm import tqdm


def constfn(val):
    def f(_):
        return val
    return f

def learn(env, total_timesteps, seed=None, nsteps=1024, ent_coef=0.01, lr=0.01,
            vf_coef=0.5,p_coef=1.0,max_grad_norm=None, gamma=0.99, lam=0.95, nminibatches=15, noptepochs=4, cliprange=0.2,
            save_interval=100,copeoperation=False,human_ent_coef=0.01,human_vf_coef=0.5,human_p_coef=1.0):

    set_global_seeds(seed)
    sess= get_session()
    global_summary=tf.summary.FileWriter('summaries/' +'feeding'+ datetime.datetime.now().strftime('%d-%m-%y%H%M'), sess.graph)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    # Get the nb of env
    nenvs = env.num_envs
    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    if copeoperation==True:
        human_model = Model(env=env, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=human_ent_coef, vf_coef=human_vf_coef,
                      p_coef=human_p_coef,
                      max_grad_norm=max_grad_norm,human=True,robot=False)
        robot_model = Model(env=env, nbatch_act=nenvs, nbatch_train=nbatch_train, ent_coef=ent_coef, vf_coef=vf_coef,
                      p_coef=p_coef,
                      max_grad_norm=max_grad_norm,human=False,robot=True)

    if copeoperation==False:
        model = Model(env=env,  nbatch_act=nenvs, nbatch_train=nbatch_train,ent_coef=ent_coef, vf_coef=vf_coef,p_coef=p_coef,
                    max_grad_norm=max_grad_norm)
    initialize()

    # Instantiate the runner object
    if copeoperation==True:
        runner = Runner(env=env, model=None, nsteps=nsteps, gamma=gamma, lam=lam,human_model=human_model,robot_model=robot_model)
    if copeoperation == False:
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=10)#recent 10 episode
    pbar=tqdm(total=total_timesteps,dynamic_ncols=True)

    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        # Start timer
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # Get minibatch
        if copeoperation==False:
            obs, returns, masks, actions, values, neglogpacs, epinfos = runner.run()
        if copeoperation == True:
            obs, human_returns, robot_returns, masks, human_actions, robot_actions, human_values, robot_values, human_neglogpacs, robot_neglogpacs, epinfos = runner.coop_run()
        epinfobuf.extend(epinfos)
        mblossvals = []
        human_mblossvals = []
        robot_mblossvals = []
        inds = np.arange(nbatch)
        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(inds)
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]
                if copeoperation == True:
                    human_slices = (arr[mbinds] for arr in (obs[:,24:], human_returns,  human_actions, human_values,  human_neglogpacs))
                    robot_slices = (arr[mbinds] for arr in
                                    (obs[:,:24], robot_returns, robot_actions, robot_values,robot_neglogpacs))
                    human_mblossvals.append(human_model.train(lrnow, cliprangenow, *human_slices))
                    robot_mblossvals.append(robot_model.train(lrnow, cliprangenow, *robot_slices))
                if copeoperation==False:
                    slices = (arr[mbinds] for arr in (obs, returns, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))#None
        # Feedforward --> get losses --> update
        if copeoperation == True:
            human_lossvals = np.mean(human_mblossvals, axis=0)
            robot_lossvals = np.mean(robot_mblossvals, axis=0)
        if copeoperation==False:
            lossvals = np.mean(mblossvals, axis=0)
        summary=tf.Summary()
        if copeoperation == True:
            human_ev = explained_variance(human_values, human_returns)
            robot_ev = explained_variance(robot_values, robot_returns)
        if copeoperation==False:
            ev = explained_variance(values, returns)
        performance_r=np.mean([epinfo['r'] for epinfo in epinfobuf])
        performance_len=np.mean([epinfo['l'] for epinfo in epinfobuf])
        success_time=np.mean([epinfo['success_time'] for epinfo in epinfobuf])
        fall_time=np.mean([epinfo['fall_time'] for epinfo in epinfobuf])
        summary.value.add(tag='Perf/Reward',simple_value=performance_r)
        summary.value.add(tag='Perf/episode_len', simple_value=performance_len)
        summary.value.add(tag='Perf/success_time', simple_value=success_time)
        summary.value.add(tag='Perf/fall_time', simple_value=fall_time)
        if copeoperation == True:
            summary.value.add(tag='Perf/human_explained_variance', simple_value=float(human_ev))
            summary.value.add(tag='Perf/robot_explained_variance', simple_value=float(robot_ev))
        if copeoperation==False:
            summary.value.add(tag='Perf/explained_variance', simple_value=float(ev))
        if copeoperation == True:
            for (human_lossval, human_lossname) in zip(human_lossvals, human_model.loss_names):
                if human_lossname=='grad_norm':
                    summary.value.add(tag='grad/' + human_lossname, simple_value=human_lossval)
                else:
                    summary.value.add(tag='human_loss/' + human_lossname, simple_value=human_lossval)
            for (robot_lossval, robot_lossname) in zip(robot_lossvals, robot_model.loss_names):
                if robot_lossname=='grad_norm':
                    summary.value.add(tag='grad/' + robot_lossname, simple_value=robot_lossval)
                else:
                    summary.value.add(tag='robot_loss/' + robot_lossname, simple_value=robot_lossval)
        if copeoperation==False:
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                if lossname=='grad_norm':
                    summary.value.add(tag='grad/' + lossname, simple_value=lossval)
                else:
                    summary.value.add(tag='loss/' + lossname, simple_value=lossval)

        global_summary.add_summary(summary,int(update*nbatch))
        global_summary.flush()
        print('finish one update')
        if update%10==0:
            msg='step: {},episode reward: {},episode len: {},success_time: {},fall_time: {}'
            pbar.update(update*nbatch)
            pbar.set_description(msg.format(update*nbatch,performance_r,performance_len,success_time,fall_time))

        if update % save_interval == 0:
            tnow = time.perf_counter()
            print('consume time', tnow - tfirststart)
            if copeoperation == True:
                savepath = osp.join("my_model_cop/", '%.5i'%update)
            if copeoperation==False:
                savepath = osp.join("my_model/", '%.5i' % update)
            os.makedirs(savepath, exist_ok=True)
            savepath = osp.join(savepath, 'ppomodel')
            print('Saving to', savepath)
            save_state(savepath)
    pbar.close()

    return model
