#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .utils import mkdir, de2bi
import numpy as np
import numpy.matlib
import scipy.io as sio
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf

# from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GRUCell
from .MIMO_detection import corr_channel, nr_corr_channel
from .utils import QAM_Modulation, indicator
from scipy.linalg import cholesky

ext_set_name = 'ext'


def train_GEPNet(test=False, trainSet=None):
    T, iter = trainSet.T, trainSet.TurboIterations
    Mr, Nt, mu, SNR = trainSet.m, trainSet.n, trainSet.mu, trainSet.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainSet.lr, trainSet.lr_decay, trainSet.decay_steps, trainSet.min_lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    tsample_size = total_batch * batch_size
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    savefile = trainSet.savefile
    net = trainSet.net
    net_loss = trainSet.loss
    directory = './model/coded/' + net + '_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    x_, y_, H_, sigma2_, label_, bits_, bs_, pp_llr_, ext_llr_ = prob.x_, prob.y_, prob.H_, prob.sigma2_,\
                                                        prob.label_, prob.bits_, prob.sample_size_, prob.pp_llr_,\
                                                        prob.ext_llr_
    model = GEPNet(trainSet=trainSet)
    loss_, ub, cavity_prob, mse, llr_e = model.build(x_, y_, H_, sigma2_, label_, bits_,
                                                     bs=bs_,pp_llr=pp_llr_,iter=iter,
                                                     ext_llr=ext_llr_)  # transfer place holder and build the model
    train = []
    global_step = tf.Variable(0, trainable=False)
    lr_ = tf.train.exponential_decay(lr, global_step, decay_steps, lr_decay, name='lr')
    grads_, _ = tf.clip_by_global_norm(tf.gradients(loss_, tf.trainable_variables()),
                                       trainSet.grad_clip)
    if trainSet.grad_clip_flag:
        optimizer = tf.train.AdamOptimizer(lr_)
        if tf.trainable_variables():
            train = optimizer.apply_gradients(zip(grads_, tf.trainable_variables()), global_step)
    else:
        if tf.trainable_variables():
            train = tf.train.AdamOptimizer(lr_).minimize(loss_, global_step, var_list=tf.trainable_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)
    if test:
        return sess, ub, cavity_prob, mse

    loss_history = []
    save = {}  # for the best model
    ivl = 5
    # generate validation set
    _, _, _, _, _, _, _, yval, xval, Hval, sigma2val, labelval, bitsval, llrval = sample_gen(trainSet, 1, vsample_size)
    llrval_sort = np.sort(abs(llrval.reshape(-1)))
    print('llr bound:', llrval_sort[int(np.ceil(0.97 * len(llrval_sort)))])
    if iter == 1:
        llrval = np.zeros((vsample_size, mu, Nt))
    else:
        if net_loss != 'ext':
            extllrv = np.zeros((vsample_size, mu, Nt))
        else:
            dataset = './dataset/' + net + ('_' + ext_set_name if trainSet.loss == 'ext' else '')+ '_valid_' + str(Mr) + 'x' + str(Nt) +\
                      '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB' + '_T' + str(T) + '_I' + str(iter) + \
                      (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + '.mat'
            if SNR == 'varying_':
                yval, xval, Hval, labelval, llrval, bitsval, extllrv, sigma2val = load_dataset(dataset, trainSet.loss,
                                                                                               snr=SNR)
            else:
                yval, xval, Hval, labelval, llrval, bitsval, extllrv = load_dataset(dataset, trainSet.loss)
                sigma2val = Nt / Mr * 10 ** (-SNR / 10) * np.ones((vsample_size, 1, 1))
            dataset = './dataset/' + net + ('_' + ext_set_name if trainSet.loss == 'ext' else '')+ '_train_' + str(Mr) + 'x' + str(Nt) +\
                      '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB' + '_T' + str(T) + '_I' + str(iter) + \
                      (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + '.mat'
            if SNR == 'varying_':
                ytrain, xtrain, Htrain, labeltrain, llrtrain, bitstrain, extllrt, sigma2train = load_dataset(dataset, trainSet.loss,
                                                                                                             snr=SNR)
            else:
                ytrain, xtrain, Htrain, labeltrain, llrtrain, bitstrain, extllrt = load_dataset(dataset, trainSet.loss)
                sigma2train = Nt / Mr * 10 ** (-SNR / 10) * np.ones((tsample_size, 1, 1))

    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch
    # loss_batch, grads = grad(xval, yval, Hval, sigma2val, labelval)  # call to generate all trainable variables

    for i in range(maxit + 1):
        if i % ivl == 0:  # validation:don't use optimizer
            loss = 0.
            # neg = 0.
            for m in range(total_batch):
                xbatch = xval[m * val_batch_size: (m + 1) * val_batch_size]
                ybatch = yval[m * val_batch_size: (m + 1) * val_batch_size]
                Hbatch = Hval[m * val_batch_size:(m + 1) * val_batch_size]
                sigma2batch = sigma2val[m * val_batch_size:(m + 1) * val_batch_size]
                labelbatch = labelval[m * val_batch_size:(m + 1) * val_batch_size]
                llrbatch = llrval[m * val_batch_size:(m + 1) * val_batch_size]

                extllrbatch = extllrv[m * val_batch_size:(m + 1) * val_batch_size]

                bitbatch = bitsval[m * val_batch_size:(m + 1) * val_batch_size]
                loss_batch, ub_batch, llre_batch = sess.run((loss_, ub, llr_e), feed_dict={y_: ybatch,
                                                        x_: xbatch, H_: Hbatch, sigma2_: sigma2batch,
                                                        bs_: val_batch_size,
                                                        label_: labelbatch,
                                                        bits_: bitbatch,
                                                        pp_llr_: llrbatch,
                                                        ext_llr_: extllrbatch})
                loss += loss_batch
            if np.isnan(loss):
                raise RuntimeError('loss is NaN')
            loss_history = np.append(loss_history, loss)
            loss_best = loss_history.min()
            # for the best model
            if loss == loss_best:
                for v in tf.trainable_variables():
                    save[str(v.name)] = sess.run(v)
                    #
            sys.stdout.write('\ri={i:<6d} loss={loss:.9f} (best={best:.9f})'
                             .format(i=i, loss=loss, best=loss_best))
            sys.stdout.flush()
            if i % (100 * 1) == 0:
                print('')


        # generate trainset
        if iter == 1:
            y, x, H, sigma2, label, bits, llr, _, _, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
            llr = np.zeros((tsample_size, mu, Nt))
        else:
            if net_loss != 'ext':
                y, x, H, sigma2, label, bits, llr, _, _, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch,
                                                                                    1)
                extllr = np.zeros((tsample_size, mu, Nt))
            # pass
            else:
                idx = i % 60
                x = xtrain[idx * tsample_size: (idx + 1) * tsample_size]
                y = ytrain[idx * tsample_size: (idx + 1) * tsample_size]
                H = Htrain[idx * tsample_size:(idx + 1) * tsample_size]
                sigma2 = sigma2train
                label = labeltrain[idx * tsample_size:(idx + 1) * tsample_size]
                bits = bitstrain[idx * tsample_size:(idx + 1) * tsample_size]
                llr = llrtrain[idx* tsample_size:(idx + 1) * tsample_size]
                extllr = extllrt[idx* tsample_size:(idx + 1) * tsample_size]


        # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
        for m in range(total_batch):
            train_loss, _, grad = sess.run((loss_, train, grads_),
                                            feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                      x_: x[m * batch_size:(m + 1) * batch_size],
                                                      H_: H[m * batch_size:(m + 1) * batch_size],
                                                      sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                      bs_: batch_size,
                                                      label_: label[m * batch_size:(m + 1) * batch_size],
                                                      bits_: bits[m * batch_size:(m+1) * batch_size],
                                                      pp_llr_: llr[m * batch_size:(m + 1) * batch_size],
                                                      ext_llr_: extllr[m * batch_size:(m + 1) * batch_size]})

    # for the best model----it's for the strange phenomenon
    tv = dict([(str(v.name), v) for v in tf.trainable_variables()])
    for k, d in save.items():
        if k in tv:
            sess.run(tf.assign(tv[k], d))
            print('restoring ' + k)
            # print('restoring ' + k + ' = ' + str(d))

    log = log + '\nloss={loss:.9f} in {i} iterations   best={best:.9f} in {j} ' \
                'iterations'.format(loss=loss, i=i, best=loss_best, j=loss_history.argmin() * ivl)

    state['done'] = done
    state['log'] = log
    save_trainable_vars(sess, savefile, **state)

    para = {}
    if net == 'GEPNet':
        for k, v in np.load(savefile).items():
            para[k.replace(':', '').replace('/', '_')] = v
    elif net == 'HyperGEPNet':
        for k, v in np.load(savefile).items():
            para[k[:k.find(':')].replace('/', '_')] = v
    para['loss_history'] = loss_history
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return


class GEPNet:
    def __init__(self, trainSet=None):
        self.T, iter = trainSet.T, trainSet.TurboIterations
        self.Nt, self.mu = trainSet.n, trainSet.mu
        size = trainSet.size
        self.GNN = GNN_GRU(n_hidden_1=size, n_hidden_2=size/2, n_gru_hidden_units=size, label_size=2**(self.mu//2), T=0,
                           ed=trainSet.ed, ed_para=trainSet.ed_para)
        if self.mu == 2:  # (+1 -1): (1 0)
            bin_order = np.array([1, 0])
            self.real_constel_norm = np.array([+1, -1]) / np.sqrt(2)
        elif self.mu == 4:  # (+3 +1 -1 -3): (10, 11 01 00)
            bin_order = np.array([2, 3, 1, 0])
            self.real_constel_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
        else:  # (+7 +5 +3 +1 -1 -3 -5 -7): (100 101 111 110 010 011 001 000)
            bin_order = np.array([4, 5, 7, 6, 2, 3, 1, 0])
            self.real_constel_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
        self.bin_array = np.sign(de2bi(bin_order, self.mu // 2) - 0.5).astype(np.float64)
        self.loss = trainSet.loss


    # @tf.Module.with_name_scope  x is no_usage
    def build(self, x, y, H, sigma2, label=tf.zeros([1], tf.float64), bits=None, bs=None, test=False, pp_llr=None,
              iter=None, ext_llr=None):
        HT = tf.transpose(H, perm=[0, 2, 1])
        HTH = tf.matmul(HT, H)
        noise_var = tf.cast(sigma2 / 2, dtype=tf.float64)

        # precompute some tensorflow constants
        eps1 = tf.constant(1e-7, dtype=tf.float64)
        eps2 = tf.constant(5e-13, dtype=tf.float64)
        ones = np.ones((1, 2*self.Nt, 2 ** (self.mu//2)), dtype=np.float64)
        pp_llr = tf.concat([pp_llr[:, :self.mu // 2, :], pp_llr[:, self.mu // 2:, :]], axis=2)  # (bs, mu//2, 2N)
        # calculate soft estimates-- mean and  variance of constellation
        dist = 0.5 * tf.matmul(self.bin_array, pp_llr)  # (bs, 2**(mu//2), 2N)
        dist += tf.reduce_min(dist, axis=1, keepdims=True)
        probs = tf.transpose(tf.exp(dist), perm=[0, 2, 1])  # (bs, 2N, 2**(mu//2))
        probs = probs / tf.reduce_sum(probs, axis=2, keepdims=True)
        s_est = tf.reduce_sum(probs * self.real_constel_norm, axis=2,
                              keepdims=True)  # (bs, 2N, 1)
        e_est = tf.reduce_sum(probs * (s_est * ones - self.real_constel_norm) ** 2,
                              axis=2, keepdims=True)  # (bs, 2N, 1)
        e_est = tf.maximum(e_est, 1e-5)
        Lambda = 1 / e_est  # (bs, 2N, 1)
        gamma = s_est * Lambda  # (bs, 2N, 1)
        beta = tf.constant(0.2, dtype=tf.float64)

        loss = 0.
        p_gnn = 0.
        vari_feat_vec, hidden_state = [], []
        mse = []
        for t in range(self.T):
            Sigma = tf.linalg.inv(HTH + noise_var * tf.linalg.diag(tf.squeeze(Lambda)))  ###
            Mu = tf.matmul(Sigma, tf.matmul(HT, y) + noise_var * gamma)  # (bs, 2N, 1)   ###

            # compute the extrinsic mean and covariance matrix
            diag = noise_var * tf.expand_dims(tf.linalg.diag_part(Sigma), -1)  # (bs, 2N, 2N) -> (bs, 2N, 1)###
            vab = tf.divide(diag, 1 - diag * Lambda)  # (bs, 2N, 1)  ###
            vab = tf.maximum(vab, eps1)
            uab = vab * (Mu / diag - gamma)  # (bs, 2N, 1)
            attr = tf.concat([uab, vab], axis=2)  # (bs, 2N, 2)

            # adjust posterior (cavity) probability distribution using GNN
            diag = diag / noise_var
            rho = Sigma ** 2 / diag / tf.transpose(diag, perm=[0, 2, 1])
            gnn_loss, p_gnn, vari_feat_vec, hidden_state = self.GNN(label, x, y, H, HTH, noise_var, bs=bs,
                                                                    u=vari_feat_vec, g_prev=hidden_state,
                                                                    attr=attr, T=t, cov=rho)
            # compute the posterior mean and covariance matrix
            p_gnn = tf.transpose(p_gnn, perm=[0, 2, 1])  # (bs, 2N, 2**(mu//2))
            post_probs = probs * p_gnn
            post_probs = post_probs / tf.reduce_sum(post_probs, axis=2 ,keepdims=True)
            ub = tf.reduce_sum(post_probs * self.real_constel_norm,
                               axis=2, keepdims=True)
            vb = tf.reduce_sum(post_probs * tf.abs(ub * ones
                                              - self.real_constel_norm) ** 2, axis=2, keepdims=True)
            vb = tf.maximum(vb, eps2)
            mse.append(tf.reduce_mean((ub - x)**2))

            # moment matching and damping
            gamma_last = gamma
            Lambda_last = Lambda
            gamma = (ub * vab - uab * vb) / vb / vab  ###
            Lambda = (vab - vb) / vb / vab  ###
            # If x and y are also provided (both have non-None values) the condition tensor acts as a mask
            # that chooses whether the corresponding element / row in the output should be taken from x
            # (if the element in condition is True) or y (if it is False).
            condition = Lambda < 0.
            Lambda = tf.where(condition=condition, x=Lambda_last, y=Lambda)
            gamma = tf.where(condition=condition, x=gamma_last, y=gamma)
            gamma = beta * gamma + (1 - beta) * gamma_last
            Lambda = beta * Lambda + (1 - beta) * Lambda_last

        if self.loss == 'mse':
            xhat = tf.reduce_sum(p_gnn * self.real_constel_norm,
                                 axis=2, keepdims=True)
            loss += tf.nn.l2_loss(x - xhat)
        else:
            loss += gnn_loss
            llr_e = calc_llr_real_tf(2 * self.Nt, self.mu, p_gnn, eps=tf.constant(1e-15, dtype=tf.float64))  # llr=log(p1/p0)
            if self.loss == 'ext':
                ext_bits = 1 / (1 + tf.exp(-ext_llr))
                soft_bits = 1 / (1 + tf.exp(-tf.transpose(llr_e, perm=[0, 2, 1])))  # soft_bits = p(b=1)
                loss = - tf.reduce_sum(ext_bits * tf.math.log(soft_bits) + (1 - ext_bits) * tf.math.log(1-soft_bits)) \
                       / (tf.cast(bs, dtype=tf.float64) * self.Nt * self.mu)
        return loss, ub, p_gnn, tf.concat([mse], axis=0), tf.transpose(llr_e, perm=[0, 2, 1])

    def calc_negentroppy(self, llr):
        mean, var = tf.reduce_mean(llr), tf.math.reduce_variance(llr)
        llr = (llr - mean) / tf.math.sqrt(var)
        k1, k2 = 36 / (8 * np.sqrt(3) - 9), 24 / (16 * np.sqrt(3) - 27)
        tmp1 = (tf.reduce_mean(llr * tf.exp(-llr ** 2 / 2))) ** 2
        tmp2 = (tf.reduce_mean(tf.exp(-llr ** 2 / 2)) - 1 / np.sqrt(2)) ** 2
        return k1 * tmp1 + k2 * tmp2


class mlp(tf.Module):
    def __init__(self, n_hidden_1=64, n_hidden_2=32, output_size=8, activation=None, name=None):
        super(mlp, self).__init__()
        # Dense: computes the dot product between the inputs and the kernel
        # along the last axis of the inputs and axis 0 of the kernel
        if name is None:
            self.dense1 = Dense(n_hidden_1, activation='relu')
            self.dense2 = Dense(n_hidden_2, activation='relu')
            self.output_layer = Dense(output_size, activation=activation)
        else:
            self.dense1 = Dense(n_hidden_1, activation='relu', name=name + '_hidden1')
            self.dense2 = Dense(n_hidden_2, activation='relu', name=name + '_hidden2')
            self.output_layer = Dense(output_size, activation=activation, name=name + '_output')

    def __call__(self, inputs):
        tmp = self.dense1(inputs)
        tmp = self.dense2(tmp)
        return self.output_layer(tmp)


class GNN_GRU:  # todo: implementation of network, size
    # n_hidden_1 = n_gru_hidden_units
    def __init__(self, n_hidden_1=64, n_hidden_2=32, n_gru_hidden_units=64, msg_size=8, label_size=4, T=0,
                 ed=None, ed_para=None):
        # with self.name_scope:
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.msg_size = msg_size
        self.n_gru_hidden_units = n_gru_hidden_units
        self.label_size = label_size
        self.ed, self.ed_para = ed, ed_para

        # factor mlp using subclass
        self.factor_mlp = mlp(output_size=msg_size, n_hidden_1=n_hidden_1,
                              n_hidden_2=n_hidden_2)  # input_size: (None, 2*msg_size+2)

        # GRU  usage: https://tensorflow.google.cn/guide/keras/rnn
        self.gru = GRUCell(units=n_gru_hidden_units)  # process a single timestep
        # (bs, input_size) --> (bs, output_size/state_size)
        # self.rnn = RNN(self.gru(1), return_state=True)  # wrap into a RNN layer
        # input: (batch_size, n_time_steps, input_size)

        # initialize single layer NN
        if T == 0:
            self.W1 = tf.Variable(tf.random.normal([3, msg_size], dtype=tf.float64),
                                  name='W1', trainable=True)
            self.b1 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b1', trainable=True)

        # variable mlp
        self.W2 = tf.Variable(tf.random.normal([n_gru_hidden_units, msg_size], dtype=tf.float64),
                              name='W2' + str(T), trainable=True)
        self.b2 = tf.Variable(tf.zeros([msg_size], dtype=tf.float64), name='b2' + str(T), trainable=True)

        # readout mlp using subclass
        self.readout_mlp = mlp(output_size=label_size, activation='softmax',
                               n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2)  # input_size: (None, msg_size)

    # @tf.Module.with_name_scope
    def __call__(self, label, x, y, h_matrix, HTH, sigma2, u, g_prev, attr, bs=None, L=2, T=0, iter=1, cov=None):
        # K = x.shape[1]
        K = h_matrix.shape[2]
        self.bs = bs
        # initial variable feature u for each variable -- variable feature vector
        if T == 0:
            ydoth = tf.matmul(tf.transpose(y, perm=[0, 2, 1]), h_matrix)  # (bs, 1, K)
            hdoth = tf.expand_dims(tf.linalg.diag_part(HTH), axis=1)  # (bs, 1, K)
            sigma2_vector = sigma2 * tf.ones_like(ydoth, dtype=tf.float64)  # (bs, 1, K)
            chan_info = tf.concat([ydoth, hdoth, sigma2_vector], axis=1)  # (bs, 3, K)
            u = tf.matmul(tf.transpose(chan_info, perm=[0, 2, 1]), self.W1)  # (bs, K, msg_size)
            u = tf.transpose(u + self.b1, perm=[0, 2, 1])  # (bs, msg_size, K)
            g_prev = tf.zeros([bs * K, self.n_gru_hidden_units], dtype=tf.float64)  # (bs * K, n_gru_hidden_units=64)

        # msg = tf.zeros([y.shape[0], K, K,], dtype=tf.float64)  # (bs, K, K, msg_size)  zero-diagonal
        for l in range(L):
            # factor to variable
            msg = self.factor2variable(K, u, HTH, sigma2, cov=cov)  # (bs, K, msg_size)
            # variable to factor
            u, g_prev = self.variable2factor(K, msg=msg,  # u: (bs, msg_size, K) g: (bs, n_gru_hidden_units*K)
                                             attr=attr, g_prev=g_prev, iter=iter)
        # readout & calculate loss
        p_gnn = tf.cast(self.readout(K, u), dtype=tf.float64)  # (bs, label_size, K)
        loss = -tf.reduce_sum(label * tf.math.log(p_gnn)) / tf.cast(bs, dtype=tf.float64)
        # feedback GRU hidden state and variable feature vector
        return loss, p_gnn, u, g_prev

    def factor2variable(self, K, u, HTH, sigma2, cov=None):  # factor2variable  K(K-1) factors
        all_input = []
        mask = []
        for k in range(K):
            u_source = tf.expand_dims(u[:, :, k], axis=1) * tf.ones([self.bs, K - 1, u.shape[1]],
                                                                    dtype=tf.float64)  # (bs, K-1, msg_size)
            u_target = tf.transpose(tf.concat([u[:, :, :k], u[:, :, k + 1:]], axis=2),
                                    perm=[0, 2, 1])  # (bs, K-1, msg_size)
            factor_feat = tf.expand_dims(tf.concat([HTH[:, k, :k], HTH[:, k, k + 1:]], axis=1), axis=-1)  # (bs, K-1, 1)
            if self.ed:
                cov_feat = tf.expand_dims(tf.concat([cov[:, k, :k], cov[:, k, k + 1:]], axis=1),
                                          axis=-1)  # (bs, K-1, 1)
                avg_rho = tf.reduce_mean(cov_feat, axis=1, keepdims=True)  # (bs, 1, 1)
                flag = tf.transpose(cov_feat >= self.ed_para * avg_rho, perm=[0, 2, 1])  # (bs, 1, K - 1)
                flag = tf.reshape(tf.repeat(flag, repeats=self.msg_size),
                                  [self.bs, 1, K - 1, self.msg_size])  # (bs, 1, K - 1, msg_size)
                mask.append(flag)
            input = tf.concat([u_source, u_target, factor_feat,
                               sigma2 * tf.ones([self.bs, K - 1, 1], dtype=tf.float64)],
                              axis=2)  # (bs, K-1, 2*msg_size+3)
            all_input.append(input)
        all_input = tf.stack(all_input, axis=1)  # (bs, K, K-1, 2*msg_size+3)
        msg = tf.cast(self.factor_mlp(tf.reshape(all_input, [-1, K, K - 1, 2 * self.msg_size + 2])),
                      dtype=tf.float64)  # (bs, K, K-1, msg_size)
        if self.ed:
            mask = tf.concat(mask, axis=1)  # (bs, K, K-1, msg_size)
            msg = tf.reduce_sum(tf.where(condition=mask, x=msg,
                                         y=tf.zeros([self.bs, K, K - 1, self.msg_size], dtype=tf.float64)), axis=2)
        else:
            msg = tf.reduce_sum(msg, axis=2)  # (bs, K, msg_size)
        return msg

    def variable2factor(self, K, msg, attr, g_prev, iter=1):  # variable2factor h:hidden vectors  K variables, share a GRU
        msg_size = self.msg_size
        batch_size = self.bs
        msg = tf.reshape(msg, [batch_size * K, msg_size])  # (bs*K, msg_size)
        if iter > 1:
            mu = 2 * int(np.log2(self.label_size))
            attr = tf.reshape(attr, [batch_size * K, 2+mu//2])  # (bs*K, attribute_size)
        else:
            attr = tf.reshape(attr, [batch_size * K, 2])  # (bs*K, attribute_size)
        input = tf.concat([msg, attr], axis=1)  # (bs*K, msg_size+attribute_size)
        _, g = self.gru(inputs=input, states=g_prev)  # g: (bs*K, n_gru_hidden_units=64)
        g = tf.cast(g, dtype=tf.float64)
        u = tf.matmul(g, self.W2) + self.b2  # (bs*K, msg_size)
        u = tf.transpose(tf.reshape(u, [batch_size, K, msg_size]), perm=[0, 2, 1])  # (bs, msg_size, K)
        return u, g

    def readout(self, K, u):
        batch_size = self.bs
        u = tf.reshape(tf.transpose(u, perm=[0, 2, 1]), [batch_size * K, self.msg_size])  # (bs*K, msg_size)
        p_gnn = self.readout_mlp(u)  # (bs*K, label_size)
        p_gnn = tf.transpose(tf.reshape(p_gnn, [batch_size, K, self.label_size]),
                             perm=[0, 2, 1])  # reverse operation: (bs, label_size, K)
        return p_gnn


def calc_llr_real_tf(N, mu, ext_probs, eps=1e-15):   # ext_probs: (bs, N=2Nt, label_size=2**(mu//2))
    if mu == 2:  # (+1 -1): (1 0)
        bin_order = np.array([1, 0])
    elif mu == 4:  # (+3 +1 -1 -3): (10, 11 01 00)
        bin_order = np.array([2, 3, 1, 0])
    else:  # (+7 +5 +3 +1 -1 -3 -5 -7): (100 101 111 110 010 011 001 000)
        bin_order = np.array([4, 5, 7, 6, 2, 3, 1, 0])
    bin_array = de2bi(bin_order, mu // 2)
    llr_real = []
    for n in range(N):
        llr_real_row = []
        for b in range(mu // 2):
            pos, neg = 0., 0.
            for z in range(2 ** (mu // 2)):
                if bin_array[z, b] == 1:
                    pos += ext_probs[:, n, z]
                else:
                    neg += ext_probs[:, n, z]
            llr_real_row.append(tf.math.log(pos + eps) - tf.math.log(neg + eps))  # column (bs,)
        llr_real.append(tf.stack(llr_real_row, axis=1))  # llr_real_row: (bs, mu//2)
    llr_real = tf.stack(llr_real, axis=1)  # llr_real: (bs, N, mu//2)
    llr = tf.concat([llr_real[:, :N // 2, :], llr_real[:, N // 2:, :]], axis=2)
    return llr  # (bs, Nt, mu)


def load_dataset(filename, loss=None, snr=None):
    import scipy.io as sio
    dataset = sio.loadmat(filename)
    H = dataset['H_dataset']
    H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=2),
                        np.concatenate((np.imag(H), np.real(H)), axis=2)), axis=1)
    label, llr, bit = dataset['label_dataset'].astype(np.float64), dataset['llr_dataset'],\
                      dataset['bit_dataset'].astype(np.float64)
    x, y = dataset['x_dataset'], dataset['y_dataset']
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x, y = np.concatenate((np.real(x), np.imag(x)), axis=1), np.concatenate((np.real(y), np.imag(y)), axis=1)
    x, y = np.expand_dims(x, axis=2), np.expand_dims(y, axis=2)
    if loss is None:
        return y, x, H, label, llr, bit
    else:
        ext_llr = dataset['ext_llr']
        if snr == 'varying_':
            sigma2 = dataset['sigma2_dataset']
            return y, x, H, label, llr, bit, ext_llr, sigma2
        return y, x, H, label, llr, bit, ext_llr


# generate extrinsic training LLR datasets
def gen_ext_llr(train=True, trainSet=None):
    T, iter = trainSet.T, trainSet.TurboIterations
    Mr, Nt, mu, SNR = trainSet.m, trainSet.n, trainSet.mu, trainSet.snr
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    vsample_size = trainSet.vsample_size
    tsample_size = 76800
    savefile = trainSet.savefile
    net = trainSet.net
    directory = './model/coded/' + net + '_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    x_, y_, H_, sigma2_, label_, bits_, bs_, pp_llr_, ext_llr_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, \
                                                       prob.label_, prob.bits_, prob.sample_size_, prob.pp_llr_, prob.ext_llr_
    model = GEPNet(trainSet=trainSet)
    loss_, ub, cavity_prob, mse, llr_e = model.build(x_, y_, H_, sigma2_, label_, bits_,
                                                     bs=bs_, pp_llr=pp_llr_, ext_llr=ext_llr_,
                                                     iter=iter)  # transfer place holder and build the model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    state = load_trainable_vars(sess, savefile)
    done = state.get('done', [])
    log = str(state.get('log', ''))
    print(log)

    if train:
        dataset = './dataset/' + net + '_train_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + \
                  str(SNR) + 'dB' + '_T' + str(T) + '_I' + str(iter) + \
                  (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + '.mat'
        # y, x, H, label, llr, bits = load_dataset(dataset)
        np.random.seed(2)
        y, x, H, sigma2, label, bits, llr, _, _, _, _, _, _, _ = sample_gen(trainSet, tsample_size, 1)
    else:
        dataset = './dataset/' + net + '_valid_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + \
                  str(SNR) + 'dB' + '_T' + str(T) + '_I' + str(iter) + \
                  (('_' + channel_type + str(rho_tx).replace('.', '')) if channel_type == 'corr' else '') + '.mat'
        # y, x, H, label, llr, bits = load_dataset(dataset)
        _, _, _, _, _, _, _, y, x, H, sigma2, label, bits, llr = sample_gen(trainSet, 1, vsample_size)

    syn_llr, syn_y, syn_x, syn_H, syn_bits, syn_sigma2 = [], [], [], [], [], []
    mul = 8
    llr_ext = np.zeros_like(llr)
    for idx in range(llr.shape[0]):
        syn_llr_ = np.matlib.repmat(llr[idx], Nt * mu, 1).reshape((Nt * mu, mu, Nt))
        syn_y_ = np.matlib.repmat(y[idx], Nt * mu, 1).reshape((Nt * mu, 2 * Mr, 1))
        syn_x_ = np.matlib.repmat(x[idx], Nt * mu, 1).reshape((Nt * mu, 2 * Nt, 1))
        syn_H_ = np.matlib.repmat(H[idx], Nt * mu, 1).reshape((Nt * mu, 2 * Mr, 2 * Nt))
        syn_bits_ = np.matlib.repmat(bits[idx], Nt * mu, 1).reshape((Nt * mu, Nt, mu))
        syn_sigma2_ = np.matlib.repmat(sigma2[idx], Nt * mu, 1).reshape((Nt * mu, 1, 1))
        for i in range(mu):
            for j in range(Nt):
                syn_llr_[i * Nt + j, i, j] = 0.

        syn_llr.append(syn_llr_.copy())
        syn_y.append(syn_y_.copy())
        syn_x.append(syn_x_.copy())
        syn_H.append(syn_H_.copy())
        syn_bits.append(syn_bits_.copy())
        syn_sigma2.append(syn_sigma2_.copy())

        sys.stdout.write('\ridx={idx}'.format(idx=idx))
        sys.stdout.flush()

        if (idx + 1) % mul == 0:
            syn_llr = np.concatenate(tuple(syn_llr), axis=0)
            syn_y, syn_x = np.concatenate(tuple(syn_y), axis=0), np.concatenate(tuple(syn_x), axis=0)
            syn_H, syn_bits = np.concatenate(tuple(syn_H), axis=0), np.concatenate(tuple(syn_bits), axis=0)
            syn_sigma2 = np.concatenate(tuple(syn_sigma2), axis=0)
            syn_label = np.zeros((Nt * mu * mul, 2 ** (mu // 2), 2 * Nt))
            x_hat_batch, le = sess.run((ub, llr_e), feed_dict={y_: syn_y, x_: syn_x, H_: syn_H, bs_: Nt * mu * mul,
                                                               sigma2_: syn_sigma2,
                                                               label_: syn_label,
                                                               bits_: syn_bits,
                                                               pp_llr_: syn_llr})
            syn_llr, syn_y, syn_x, syn_H, syn_bits, syn_sigma2 = [], [], [], [], [], []
            for m in range(mul):
                for i in range(mu):
                    for j in range(Nt):
                        llr_ext[idx - mul + m + 1, i, j] = le[m * mu * Nt + i * Nt + j, i, j]

    # llr_ext -= llr

    import scipy.io as sio
    H = H[:, :Mr, :Nt] + 1j * H[:, Mr:2 * Mr, :Nt]
    x = x[:, :Nt, :] + 1j * x[:, Nt:2 * Nt, :]
    y = y[:, :Mr, :] + 1j * y[:, Mr:2 * Mr, :]
    sio.savemat(dataset.replace('train', ext_set_name + '_train').replace('valid', ext_set_name + '_valid'),
                {'H_dataset': H, 'bit_dataset': bits,
                 'label_dataset': label, 'llr_dataset': llr,
                 'x_dataset': x, 'y_dataset': y, 'ext_llr': llr_ext,
                 'sigma2_dataset': sigma2})
    return


def sample_gen(trainSet, ts, vs, training_flag=True):
    Mr, Nt = trainSet.m, trainSet.n
    mu, SNR = trainSet.mu, trainSet.snr
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    if training_flag is False:
        ts = 0
    # generate training samples:
    H_ = np.zeros((2 * ts * Mr, 2 * Nt))
    x_ = np.zeros((2 * ts * Nt, 1))
    y_ = np.zeros((2 * ts * Mr, 1))
    sigma2_ = np.zeros((ts, 1))
    bits_ = np.zeros((ts, Nt, mu), dtype=int)
    indicator_ = np.zeros((ts, 2 ** (mu // 2), 2 * Nt), dtype=int)
    llr_ = np.zeros((ts, mu, Nt))
    # generate development samples:
    Hval_ = np.zeros((2 * vs * Mr, 2 * Nt))
    xval_ = np.zeros((2 * vs * Nt, 1))
    yval_ = np.zeros((2 * vs * Mr, 1))
    sigma2val_ = np.zeros((vs, 1))
    bitsval_ = np.zeros((vs, Nt, mu), dtype=int)
    indicatorval_ = np.zeros((vs, 2 ** (mu // 2), 2 * Nt), dtype=int)
    llrval_ = np.zeros((vs, mu, Nt))

    # IA = [0 0.33 0.67 0.78 0.89 0.94 0.99 1]  add:0.16-->0.5
    mu_set = np.array([0., 1.2, 3.5, 4.8, 7.2, 9.4, 15.9])

    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if channel_type == 'nr':
        rspat = nr_corr_channel(Mr, Nt, 'meda', a=0.0)
        rherm = cholesky(rspat).T

    for i in range(ts + vs):
        if trainSet.snr == 'varying_':
            SNR = np.random.randint(low=9, high=16)
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)

        # generate prior LLR for bits  Normal((2d - 1)*mu, 2*mu)
        idx = np.random.randint(low=0, high=5, size=(mu, Nt))
        # idx = np.random.randint(low=0, high=2)
        # idx = 1
        llr = (2 * bits.reshape((mu, Nt), order='F') - 1) * mu_set[idx] + np.sqrt(2 * mu_set[idx]) * np.random.randn(mu, Nt)

        # Rayleigh MIMO channel
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt) +
                                   1j * np.random.randn(Mr, Nt))
        if channel_type == 'corr':  # correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
            # H = H * np.sqrt(Nt) / la.norm(H, 'fro')
        if channel_type == 'nr':
            H = np.reshape(rherm @ H.reshape(-1, 1, order='F'), (Mr, Nt), order='F')
        # channel input & output
        y = H @ x
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-SNR / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        if i < ts:
            H_[2 * Mr * i:2 * Mr * (i + 1)] = H
            x_[2 * Nt * i:2 * Nt * (i + 1)] = x
            y_[2 * Mr * i:2 * Mr * (i + 1)] = y
            bits_[i] = bits.reshape(Nt, mu)
            sigma2_[i] = sigma2
            indicator_[i] = indicator(bits, mu)
            llr_[i] = llr
        else:
            Hval_[2 * Mr * (i - ts):2 * Mr * (i - ts + 1)] = H
            xval_[2 * Nt * (i - ts):2 * Nt * (i - ts + 1)] = x
            yval_[2 * Mr * (i - ts):2 * Mr * (i - ts + 1)] = y
            bitsval_[i - ts] = bits.reshape(Nt, mu)
            sigma2val_[i - ts] = sigma2
            indicatorval_[i - ts] = indicator(bits, mu)
            llrval_[i - ts] = llr

    # reshape
    H_ = H_.reshape(ts, 2 * Mr, 2 * Nt)
    x_ = x_.reshape(ts, 2 * Nt, 1)
    y_ = y_.reshape(ts, 2 * Mr, 1)
    sigma2_ = sigma2_.reshape(ts, 1, 1)
    Hval_ = Hval_.reshape(vs, 2 * Mr, 2 * Nt)
    xval_ = xval_.reshape(vs, 2 * Nt, 1)
    yval_ = yval_.reshape(vs, 2 * Mr, 1)
    sigma2val_ = sigma2val_.reshape(vs, 1, 1)

    return y_, x_, H_, sigma2_, indicator_, bits_, llr_,\
               yval_, xval_, Hval_, sigma2val_, indicatorval_, bitsval_, llrval_
