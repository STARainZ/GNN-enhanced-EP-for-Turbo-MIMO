#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import numpy.linalg as la
import math
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# import tensorflow as tf


class Generator_(object):
    def __init__(self, A, **kwargs):
        self.A = A
        M, N = A.shape
        vars(self).update(kwargs)  # update keyword arguments
        self.H_ = tf.placeholder(tf.float64, (None, M, N), name='H')
        if self.net == 'MMNet':
            self.x_ = tf.placeholder(tf.float64, (None, N), name='x')
            self.y_ = tf.placeholder(tf.float64, (None, M), name='y')
            self.sigma2_ = tf.placeholder(tf.float64, (None, 1),
                                          name='sigma2')
        else:
            self.x_ = tf.placeholder(tf.float64, (None, N, 1), name='x')
            self.y_ = tf.placeholder(tf.float64, (None, M, 1), name='y')
            self.sigma2_ = tf.placeholder(tf.float64, (None, 1, 1),
                                          name='sigma2')  # the dimensions of dense should be defined
        # if self.coding:
        self.bits_ = tf.placeholder(tf.float64, (None, N // 2, self.mu), name='bits')
        self.info_bits_ = tf.placeholder(tf.float64, name='info_bits')
        self.label_ = tf.placeholder(tf.float64, (None, 2 ** (self.mu // 2), N), name='label')
        self.sample_size_ = tf.placeholder(tf.int32, name='sample_size')
        self.pp_llr_ = tf.placeholder(tf.float64, (None, self.mu, N // 2), name='pp_llr')
        self.ext_llr_ = tf.placeholder(tf.float64, (None, self.mu, N // 2), name='ext_llr')

        self.rw_inv_ = tf.placeholder(tf.float64, (None, M, M), name='rw_inv')
        # for training
        self.eigvalue_ = tf.placeholder(tf.float32, (None, M // 2, 1), name='eig')

        # for testing
        self.C_ = tf.placeholder(tf.complex64, (None, M // 2, M // 2), name='C')


class TFGenerator_(Generator_):
    def __init__(self, **kwargs):
        Generator_.__init__(self, **kwargs)

    def __call__(self, sess):
        """generates y,x pair for training"""
        return sess.run((self.ygen_, self.xgen_, self.Hgen_, self.sigma2gen_))


def SISO_OFDM_detection_problem(K):
    prob = TFGenerator_(A=np.zeros((2 * K, 2 * K)))
    prob.name = 'SISO_OFDM detection'
    return prob


def MIMO_detection_problem(M, N, mu, coding, net):
    prob = TFGenerator_(A=np.zeros((2 * M, 2 * N)), mu=mu, coding=coding, net=net)
    prob.name = 'MIMO detection'
    return prob
