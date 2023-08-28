#!/usr/bin/python

import numpy as np
import scipy.io as sio

para = {}
Mr = 4
Nt = 4
mu = 4
SNR = 18
loss = 'final'
coding = True
T = 5
turbo_iter = 2
net = 'EPNet'
directory = './' + ('coded/' if coding else '') + net + '_' + str(Mr) + 'x' + str(Nt) + '_' + \
            str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
load_filename = directory + '/' + net + '_' + str(Mr) + 'x' + str(Nt) + '_' + \
                str(2 ** mu) + 'QAM_' + str(SNR) + 'dB_' + \
                loss + '_T' + str(T) + \
                (('_I' + str(turbo_iter)) if turbo_iter > 1 else '') + \
                ('_coding' if coding else '') + '_rnd_bce.npz'
if net == 'GEPNet':
    for k, v in np.load(load_filename).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(load_filename.replace('.npz', '.mat'), para)
elif net == 'GEPNetLD':
    for k, v in np.load(load_filename).items():
        para[k[:k.find(':')].replace('/', '_')] = v
    sio.savemat(load_filename.replace('.npz', '.mat'), para)
elif net == 'EPNet':
    for k, v in np.load(load_filename).items():
        para[k] = v
    beta = np.zeros(20)
    for t in range(20):
        if para.get("beta_" + str(t) + ":0", -1) != -1:
            beta[t] = para["beta_" + str(t) + ":0"]
        else:
            break
    beta = 1. / (1. + np.exp(-beta[:t]))
    sio.savemat(load_filename.replace('.npz', '.mat'), {'beta': beta})
