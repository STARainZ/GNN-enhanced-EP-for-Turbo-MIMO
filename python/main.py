#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from tools import networks, problems, networks_soft

import numpy as np
import scipy.io as sio
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

np.random.seed(1)  # numpy is good about making repeatable output
tf.set_random_seed(1)

# system parameters
SNR_train = np.arange(0, 45, 5)  # SNR list for 16QAM
# SNR_train = [6, 9, 12, 15, 18]  # SNR list for QPSK
test = False  # test or train

BER = []
SER = []
FER = []
BLER = []
other_stats = []
prob = []
x_hat_T = []
sess = []
ext_probs = []
SNR_test = []


class SysIn(object):  # for testing GEPNet in uncoded systems
    mu = 4  # modulation order:2^mu QPSK:mu=2 16QAM:mu=4
    Mr = 4  # number of receiving antennas
    Nt = 4  # number of transmitting antennas
    T = 5  # number of GEPNet layers
    TurboIterations = 1
    MaxNumTrial, MinNumTrial = 5000, 500
    detect_type = 'GEPNet'  # 'EP_real'
    channel_type = ''  # channel type: 'rayleigh', 'corr' or 'winner'
    rho_tx, rho_rx = 0.5, 0.5
    sess, prob, x_hat_net, ext_probs = [], [], [], []
    batch_size = 128
    csi, gamma, Np, orth = 0, 0.95, Nt, False  # 0: perfect csi; 1: add awgn; 2: channel estimation
    savefile = 'Results_' + detect_type + \
               '_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM' + '_T' + str(T) + \
               (('_I' + str(TurboIterations)) if TurboIterations > 1 else '') + \
               (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + \
               (('_awgn' + str(gamma).replace('.', '')) if csi == 1 else '') + \
               ('_ce' if csi == 2 else '') + \
               ('_north' if (csi == 2 and orth is False) else '') + ''


class TrainSetting(object):
    mu = 4
    m, n = 4, 4
    T = 5  # number of GEPNet layers
    TurboIterations = 2  # for training the uncoded detector, set this to 1; else keeping with 2
    lr, lr_decay, min_lr, decay_steps = 1e-3, 0.1, 1e-4, 50000  # totally 50000 steps, decay to 1e-4
    maxit = 5000
    sample_size, vsample_size = 1280, 6000
    total_batch = 10
    batch_size = int(sample_size / total_batch)
    snr = 'varying_'
    channel_type, rho_tx, rho_rx = '', 0.5, 0.5
    loss = 'final'  # 'final' for Step 1 (training APP-GEPNet) and Step 2 (generating extrinsic training LLRs); 'ext' for Step 3 (train EXT-GEPNet)
    grad_clip_flag = True
    grad_clip = 200.0  # hyperparameter
    coding = True
    net = 'GEPNet'
    torch = False  # if using the model aligning with that at https://github.com/GNN-based-MIMO-Detection/GNN-based-MIMO-Detection (default: False)
    ed = False  # edge pruning
    ed_para = 0.4  # edge pruning parameter
    size = 64  # hidden layer size of GNN
    csi, gamma, Np, orth = 0, 0.95, n, False  # 0: perfect csi; 1: add awgn; 2: channel estimation
    wlmmse = None
    savefile = net + '_' + str(m) + 'x' + str(n) + '_' + str(2 ** mu) + 'QAM_' + str(snr) + 'dB_' + loss + '_T' + \
               str(T) + (('_I' + str(TurboIterations)) if TurboIterations > 1 else '') + \
               (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + \
               (('_' + channel_type) if channel_type == 'nr' else '') + \
               ('_coding' if coding else '') + ('_ed' + str(ed_para).replace('.', '') if ed else '') + \
               (('_awgn' + str(gamma).replace('.', '')) if csi == 1 else '') + \
               ('_ce' if csi == 2 else '') + ('_north' if (csi == 2 and orth is False) else '') + \
               ('_rnd' if coding else '_ini') + \
               '_damping03' + ('_torch' if torch else '') + '.npz'


sysIn = SysIn()
trainSet = TrainSetting()

# create the basic problem structure
# trainSet.prob = problems.MIMO_detection_problem(trainSet.m, trainSet.n, trainSet.mu, trainSet.coding)
trainSet.prob = problems.MIMO_detection_problem(trainSet.m, sysIn.Nt, trainSet.mu, trainSet.coding, trainSet.net)

# test
if test:
    model = []
    from tools import MIMO_detection
    if sysIn.detect_type == 'GEPNet':
        if trainSet.coding:
            sysIn.sess, sysIn.x_hat_net, sysIn.ext_probs, sysIn.mse = networks_soft.train_GEPNet(test=True,
                                                                                                 trainSet=trainSet)
        else:
            sysIn.sess, sysIn.x_hat_net, sysIn.ext_probs, sysIn.mse = networks.train_GEPNet(test=True,
                                                                                            trainSet=trainSet)
        sysIn.prob = trainSet.prob

    for i in range(1, 6):
        print("SNR=", SNR_train[i])
        SNR_test.append(SNR_train[i])
        np.random.seed(1)
        if sysIn.detect_type == 'GEPNet':
            ber, ser, stats = MIMO_detection.MIMO_detection_batch(sysIn, SNR_train[i],
                                                                  batch_size=sysIn.batch_size)
        else:
            ber, ser, stats = MIMO_detection.MIMO_detection_simulate(model, sysIn, SNR_train[i])
        SER.append(ser)
        BER.append(ber)
        other_stats.append(stats)

# train
else:
    # ***  Step 2: Generate extrinsic training LLRs (dataset) -- uncomment the following line  *** #
    # networks_soft.gen_ext_llr(train=False, trainSet=trainSet)
    if trainSet.coding:  # ***  Step 3: Train EXT-GEPNet  *** #
        networks_soft.train_GEPNet(trainSet=trainSet)
    else:  # ***  Step 1: Train APP-GEPNet  *** #
        networks.train_GEPNet(trainSet=trainSet)

print('BER', BER)
print('SER', SER)
results = np.array([BER, SER])
other_stats = np.array(other_stats).T

# save the BER results -- example
if test:
    sio.savemat(sysIn.savefile + '.mat', {sysIn.savefile: results, 'other_stats': other_stats, 'SNR': SNR_test})
