import numpy as np
import tensorflow as tf
import random


def get_session(config=None):
    sess = tf.get_default_session()
    if sess is None:
        sess =tf.InteractiveSession(config=config)
    return sess

def set_global_seeds(i):
    myseed = i if i is not None else None
    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def tile_images(img_nhwc):
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def save_state(fname):
    saver = tf.train.Saver()
    sess=get_session()
    saver.save(sess, fname)

ALREADY_INITIALIZED = set()

def initialize():
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

def explained_variance(ypred,y):
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary