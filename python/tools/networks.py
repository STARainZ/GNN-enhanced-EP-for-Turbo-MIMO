#!/usr/bin/python
from __future__ import division
from __future__ import print_function
from .train import load_trainable_vars, save_trainable_vars
from .MIMO_detection import sample_gen
from .utils import mkdir
import numpy as np
import sys
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import scipy.io as sio

# from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, GRUCell


def train_GEPNet(test=False, trainSet=None):
    Mr, Nt, mu, SNR = trainSet.m, trainSet.n, trainSet.mu, trainSet.snr
    lr, lr_decay, decay_steps, min_lr, maxit = trainSet.lr, trainSet.lr_decay, trainSet.decay_steps, trainSet.min_lr, trainSet.maxit
    vsample_size = trainSet.vsample_size
    total_batch, batch_size = trainSet.total_batch, trainSet.batch_size
    savefile = trainSet.savefile
    directory = './model/' + 'GEPNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
    mkdir(directory)
    savefile = directory + '/' + savefile
    prob = trainSet.prob
    x_, y_, H_, sigma2_, label_, bs_, rw_inv_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.label_,\
                                                prob.sample_size_, prob.rw_inv_
    model = GEPNet(trainSet=trainSet)
    loss_, ub, cavity_prob, mse = model.build(x_, y_, H_, sigma2_, label_,
                                              bs=bs_, rw_inv=rw_inv_)  # transfer place holder and build the model
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
    _, _, _, _, _, _, yval, xval, Hval, sigma2val, labelval, rw_inv_val = sample_gen(trainSet, 1, vsample_size)
    # sigma2val = Nt / Mr * 10 ** (-SNR / 10)
    val_batch_size = vsample_size // total_batch
    # loss_batch, grads = grad(xval, yval, Hval, sigma2val, labelval)  # call to generate all trainable variables

    for i in range(maxit + 1):
        if i % ivl == 0:  # validation:don't use optimizer
            loss = 0.
            for m in range(total_batch):
                xbatch = xval[m * val_batch_size: (m + 1) * val_batch_size]
                ybatch = yval[m * val_batch_size: (m + 1) * val_batch_size]
                Hbatch = Hval[m * val_batch_size:(m + 1) * val_batch_size]
                sigma2batch = sigma2val[m * val_batch_size:(m + 1) * val_batch_size]
                labelbatch = labelval[m * val_batch_size:(m + 1) * val_batch_size]
                rw_inv_batch = rw_inv_val[m * val_batch_size:(m + 1) * val_batch_size]
                loss_batch = sess.run(loss_, feed_dict={y_: ybatch,
                                                        x_: xbatch, H_: Hbatch, sigma2_: sigma2batch,
                                                        bs_: val_batch_size,
                                                        label_: labelbatch,
                                                        rw_inv_: rw_inv_batch})
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
        y, x, H, sigma2, label, rw_inv, _, _, _, _, _, _ = sample_gen(trainSet, batch_size * total_batch, 1)
        # sigma2 = Nt / Mr * 10 ** (-SNR / 10)
        for m in range(total_batch):
            train_loss, _, grad = sess.run((loss_, train, grads_),
                                           feed_dict={y_: y[m * batch_size:(m + 1) * batch_size],
                                                      x_: x[m * batch_size:(m + 1) * batch_size],
                                                      H_: H[m * batch_size:(m + 1) * batch_size],
                                                      sigma2_: sigma2[m * batch_size:(m + 1) * batch_size],
                                                      bs_: batch_size,
                                                      label_: label[m * batch_size:(m + 1) * batch_size],
                                                      rw_inv_: rw_inv[m * batch_size:(m + 1) * batch_size]})

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
    for k, v in np.load(savefile).items():
        para[k.replace(':', '').replace('/', '_')] = v
    sio.savemat(savefile.replace('.npz', '.mat'), para)

    return


class GEPNet:
    def __init__(self, trainSet=None):
        self.T, iter = trainSet.T, trainSet.TurboIterations - 1
        self.Nt, self.mu = trainSet.n, trainSet.mu
        size = trainSet.size
        self.torch = trainSet.torch  # if using the model aligning with that at https://github.com/GNN-based-MIMO-Detection/GNN-based-MIMO-Detection (problematic)
        self.coop = trainSet.coop
        self.GNN = GNN_GRU(n_hidden_1=size, n_hidden_2=size / 2, n_gru_hidden_units=size,
                           label_size=2 ** (self.mu // 2), T=0, ed=trainSet.ed, ed_para=trainSet.ed_para,
                           torch=self.torch)
        if self.mu == 2:
            self.real_constel_norm = np.array([+1, -1]) / np.sqrt(2)
        elif self.mu == 4:
            self.real_constel_norm = np.array([+3, +1, -1, -3]) / np.sqrt(10)
        else:
            self.real_constel_norm = np.array([+7, +5, +3, +1, -1, -3, -5, -7]) / np.sqrt(42)
        self.loss = trainSet.loss
        self.modified = trainSet.modified

    # @tf.Module.with_name_scope
    def build(self, x, y, H, sigma2, label=tf.zeros([1], tf.float64), bs=None, test=False, rw_inv=None):
        HT = tf.transpose(H, perm=[0, 2, 1])
        HTH = tf.matmul(HT, H)
        noise_var = tf.cast(sigma2 / 2, dtype=tf.float64)

        # precompute some tensorflow constants
        eps1 = tf.constant(1e-7, dtype=tf.float64)
        eps2 = tf.constant(5e-13, dtype=tf.float64)
        Lambda = 1 / (0.5 * tf.ones_like(x, dtype=tf.float64))
        gamma = tf.zeros_like(x, tf.float64)
        beta = tf.constant(0.3, dtype=tf.float64)

        loss = 0.
        p_gnn = 0.
        vari_feat_vec, hidden_state = [], []
        mse = []
        for t in range(self.T):
            # compute the mean and covariance matrix
            # (bs, 2N, 2N)  lambda:(bs, 2N, 1) -> (bs, 2N, 2N)
            if self.modified:
                Sigma = tf.linalg.inv(tf.matmul(tf.matmul(HT, rw_inv), H) + tf.linalg.diag(tf.squeeze(Lambda)))
                Mu = tf.matmul(Sigma, (tf.matmul(HT, tf.matmul(rw_inv, y)) + gamma))
            else:
                Sigma = tf.linalg.inv(HTH + noise_var * tf.linalg.diag(tf.squeeze(Lambda)))  ###
                Mu = tf.matmul(Sigma, tf.matmul(HT, y) + noise_var * gamma)  # (bs, 2N, 1)   ###

            # compute the extrinsic mean and covariance matrix
            if self.modified:
                diag = tf.expand_dims(tf.linalg.diag_part(Sigma), -1)
            else:
                diag = noise_var * tf.expand_dims(tf.linalg.diag_part(Sigma), -1)  # (bs, 2N, 2N) -> (bs, 2N, 1)###
            vab = tf.divide(diag, 1 - diag * Lambda)  # (bs, 2N, 1)  ###
            vab = tf.maximum(vab, eps1)
            uab = vab * (Mu / diag - gamma)  # (bs, 2N, 1)
            attr = tf.concat([uab, vab], axis=2)  # (bs, 2N, 2)

            # adjust posterior (cavity) probability distribution using GNN
            # correlation coefficient of x: rho = Sigma.^2 ./ diag(Sigma) ./ diag(Sigma).T
            diag = diag / noise_var
            rho = Sigma ** 2 / diag / tf.transpose(diag, perm=[0, 2, 1])
            gnn_loss, p_gnn, vari_feat_vec, hidden_state = self.GNN(label, x, y, H, - HTH if self.torch else HTH,
                                                                    noise_var, bs=bs,
                                                                    u=vari_feat_vec, g_prev=hidden_state,
                                                                    attr=attr, T=t, cov=rho)

            p_gnn = tf.transpose(p_gnn, perm=[0, 2, 1])
            ub = tf.reduce_sum(p_gnn * self.real_constel_norm,
                               axis=2, keepdims=True)
            vb = tf.reduce_sum(p_gnn * tf.abs(ub * np.ones((1, x.shape[1], 2 ** (self.mu // 2)), dtype=np.float64)
                                              - self.real_constel_norm) ** 2, axis=2, keepdims=True)
            vb = tf.maximum(vb, eps2)
            mse.append(tf.reduce_mean((ub - x) ** 2))

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
            if self.loss == 'avg':
                loss += gnn_loss / self.T
            elif self.loss == 'weighted':
                loss += gnn_loss * (t + 1) / self.T
        if self.loss == 'final':
            loss += gnn_loss
        return loss, ub, p_gnn, tf.concat([mse], axis=0)


class mlp(tf.Module):
    def __init__(self, n_hidden_1=64, n_hidden_2=32, output_size=8,
                 activation1=None, activation2=None, activation3=None):
        super(mlp, self).__init__()
        # Dense: computes the dot product between the inputs and the kernel
        # along the last axis of the inputs and axis 0 of the kernel
        self.dense1 = Dense(n_hidden_1, activation=activation1)
        self.dense2 = Dense(n_hidden_2, activation=activation2)
        self.output_layer = Dense(output_size, activation=activation3)

    def __call__(self, inputs, coop=False):
        tmp = self.dense1(inputs)
        tmp = self.dense2(tmp)
        return self.output_layer(tmp)


class GNN_GRU:
    def __init__(self, n_hidden_1=64, n_hidden_2=32, n_gru_hidden_units=64, msg_size=8, label_size=4, T=0,
                 ed=None, ed_para=None, torch=False):
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.msg_size = msg_size
        self.n_gru_hidden_units = n_gru_hidden_units
        self.label_size = label_size
        self.ed, self.ed_para = ed, ed_para

        # factor mlp using subclass # input_size: (None, 2*msg_size+3)
        self.factor_mlp = mlp(output_size=msg_size, n_hidden_1=n_hidden_1,
                              n_hidden_2=n_hidden_2, activation1='relu',
                              activation2='relu', activation3='relu' if torch else None)

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
        self.readout_mlp = mlp(output_size=label_size, activation1=None if torch else 'relu',
                               activation2=None if torch else 'relu', activation3='softmax',
                               n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2)  # input_size: (None, msg_size)

    # @tf.Module.with_name_scope
    def __call__(self, label, x, y, h_matrix, HTH, sigma2, u, g_prev, attr, bs=None, L=2, T=0,
                 cov=None):
        K = x.shape[1]
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
                                             attr=attr, g_prev=g_prev)
        # readout & calculate loss
        p_gnn = self.readout(K, u)  # (bs, label_size, K)
        p_gnn = tf.cast(p_gnn, dtype=tf.float64)
        loss = - tf.reduce_sum(label * tf.math.log(p_gnn)) / tf.cast(bs, dtype=tf.float64)
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
            if self.ed:  # edge pruning version
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

    def variable2factor(self, K, msg, attr, g_prev):  # variable2factor h:hidden vectors  K variables, share a GRU
        msg_size = self.msg_size
        batch_size = self.bs
        msg = tf.reshape(msg, [batch_size * K, msg_size])  # (bs*K, msg_size)
        attr = tf.reshape(attr, [batch_size * K, 2])  # (bs*K, 2)
        input = tf.concat([msg, attr], axis=1)  # (bs*K, msg_size+2)
        # input = msg
        _, g = self.gru(inputs=input, states=g_prev)  # g: (bs*K, n_gru_hidden_units=64)
        g = tf.cast(g, dtype=tf.float64)
        u = tf.matmul(g, self.W2) + self.b2  # (bs*K, msg_size)
        u = tf.transpose(tf.reshape(u, [batch_size, K, msg_size]), perm=[0, 2, 1])  # (bs, msg_size, K)
        return u, g

    def readout(self, K, u):
        batch_size = self.bs
        u = tf.reshape(tf.transpose(u, perm=[0, 2, 1]), [batch_size * K, self.msg_size])  # (bs*K, msg_size)
        p_gnn = self.readout_mlp(u, coop=True)  # (bs*K, label_size)
        p_gnn = tf.transpose(tf.reshape(p_gnn, [batch_size, K, self.label_size]),
                             perm=[0, 2, 1])  # reverse operation: (bs, label_size, K)
        return p_gnn

