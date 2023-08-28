#!/usr/bin/python
from __future__ import division
import numpy as np
import time
import numpy.linalg as la
from scipy.linalg import toeplitz, sqrtm, cholesky, dft
from .utils import QAM_Modulation, QAM_Demodulation, indicator, lmmse_ce
from .EP import EP_real_v3
from gurobipy import *

pi = math.pi


def MIMO_detection_simulate(model, sysin, SNR=40):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    csi = sysin.csi
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if csi == 2:
        Np = sysin.Np
        wlmmse, xp, rw = channel_est(sysin, SNR, orth=sysin.orth)
        sysin.rw_inv = None
    else:
        sysin.rw_inv = None

    norm = np.sqrt(1)
    # num_trail = 10000
    # for i in range(num_trail):
    while True:
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1) / norm

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        # add AWGN noise
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        sta = time.time()
        x_hat, MSE = detector(sysin, H, x, y, sigma2, MSE, model)
        end = time.time()

        # back into np.complex64
        x_hat = x_hat.reshape((2, Nt))
        x_hat = x_hat[0, :] + 1j * x_hat[1, :]

        # Demodulate
        x_hat_demod = QAM_Demodulation(x_hat*norm, mu)

        total_time += (end - sta)

        # calculate BER
        err_bits = np.sum(np.not_equal(x_hat_demod, bits))
        total_err_bits += err_bits
        total_bits += mu * Nt
        count = count + 1
        if err_bits > 0:
            ser += calc_ser(x_hat_demod, bits, Nt, mu)
            sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f}'
                             .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                                     SER=ser/count/Nt))
            sys.stdout.flush()
        if total_err_bits > err_bits_target or total_bits > 1e8:
            end = time.time()
            iter_time = end - start
            print("\nSNR=", SNR, "iter_time:", iter_time)
            ber = total_err_bits / total_bits
            ser = ser / count / Nt
            print("BER:", ber)
            print("SER:", ser)
            print("MSE:", 10 * np.log10(MSE / count))
            break

    # print("time:",total_time/1000)
    return ber, ser, np.array([total_err_bits, total_bits])


def MIMO_detection_batch(sysin, SNR=40, batch_size=64):
    Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    sess, prob, x_hat_net, mse = sysin.sess, sysin.prob, sysin.x_hat_net, sysin.mse
    csi = sysin.csi
    x_, y_, H_, sigma2_, label_, bs_, rw_inv_ = prob.x_, prob.y_, prob.H_, prob.sigma2_, prob.label_, \
                                                prob.sample_size_, prob.rw_inv_
    err_bits_target = 1000
    total_err_bits = 0
    total_bits = 0
    ser = 0
    count = 0
    start = time.time()
    MSE = 0
    total_time = 0.
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if csi == 2:
        Np = Nt
        wlmmse, xp, rw = channel_est(sysin, SNR, orth=sysin.orth)
        sysin.rw_inv = np.zeros((1, 2 * Mr, 2 * Mr))
    else:
        sysin.rw_inv = np.zeros((1, 2 * Mr, 2 * Mr))
    norm = np.sqrt(1)
    H_batch = np.zeros((batch_size, 2 * Mr, 2 * Nt))
    x_batch = np.zeros((batch_size, 2 * Nt, 1))
    y_batch = np.zeros((batch_size, 2 * Mr, 1))
    bits_batch = np.zeros((batch_size, Nt * mu), dtype=int)

    # num_trail = 10000
    # for i in range(num_trail):
    while True:
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1) / norm

        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx

        # channel input & output
        y = H @ x

        # add AWGN noise
        signal_power = Nt / Mr  # signal power per receive ant.: E(|xi|^2)=1 E(||hi||_F^2)=Nt
        sigma2 = signal_power * 10 ** (-(SNR) / 10)  # noise power per receive ant.; average SNR per receive ant.
        noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, 1)
                                       + 1j * np.random.randn(Mr, 1))
        y = y + noise

        if csi == 2:  # channel estimation
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        H_batch[count % batch_size] = H
        x_batch[count % batch_size] = x
        y_batch[count % batch_size] = y
        bits_batch[count % batch_size] = bits

        count = count + 1
        if count % batch_size == 0:
            xbatch = x_batch
            ybatch = y_batch
            sta = time.time()
            rw_inv = sysin.rw_inv
            x_hat_batch, mse_batch = sess.run((x_hat_net, mse), feed_dict={y_: ybatch,
                                                   x_: xbatch, H_: H_batch, bs_: batch_size,
                                                   sigma2_: sigma2*(np.ones((batch_size, 1, 1), dtype=np.float64)),
                                                   label_: np.zeros((batch_size, 2**(mu//2), 2*Nt), dtype=np.float64),
                                                                           rw_inv_:rw_inv})
            # x_hat: (bs, 2*Nt, 1)
            end = time.time()
            # mse = np.mean(abs(x_hat_batch - x_batch) ** 2)
            # MSE += np.array([mse])
            mse_batch = np.array(mse_batch)
            MSE += mse_batch

            for m in range(batch_size):
                x_hat = x_hat_batch[m]
                # back into np.complex64
                x_hat = x_hat.reshape((2, Nt))
                x_hat = x_hat[0, :] + 1j * x_hat[1, :]

                # Demodulate
                x_hat_demod = QAM_Demodulation(x_hat*norm, mu)

                total_time += (end - sta)

                # calculate BER
                err_bits = np.sum(np.not_equal(x_hat_demod, bits_batch[m]))
                total_err_bits += err_bits
                total_bits += mu * Nt

                if err_bits > 0:
                    ser += calc_ser(x_hat_demod, bits_batch[m], Nt, mu)
                    sys.stdout.write('\rtotal_err_bits={teb} total_bits={tb} BER={BER:.9f} SER={SER:.6f}'
                                     .format(teb=total_err_bits, tb=total_bits, BER=total_err_bits / total_bits,
                                             SER=ser / (count-batch_size+m+1) / Nt))
                    sys.stdout.flush()

            if total_err_bits > err_bits_target or total_bits > 1e7:
                end = time.time()
                iter_time = end - start
                print("\nSNR=", SNR, "iter_time:", iter_time)
                ber = total_err_bits / total_bits
                ser = ser / count / Nt
                print("BER:", ber)
                print("SER:", ser)
                print("MSE:", 10 * np.log10(MSE / count * batch_size))
                break

    return ber, ser, np.array([total_err_bits, total_bits])


def calc_ser(x_hat_demod, bits, Nt, mu):
    # ser = 0
    # for n in range(Nt):
    #     err_bits = np.sum(np.not_equal(x_hat_demod[n*mu:(n+1)*mu], bits[n*mu:(n+1)*mu]))
    #     if err_bits > 0:
    #         ser += 1
    err = np.not_equal(x_hat_demod, bits).reshape(Nt, mu)
    ser = np.sum(np.any(err, axis=1))
    return ser


def sample_gen(trainSet, ts, vs, training_flag=True):
    Mr, Nt = trainSet.m, trainSet.n
    mu, SNR = trainSet.mu, trainSet.snr
    channel_type, rho_tx, rho_rx = trainSet.channel_type, trainSet.rho_tx, trainSet.rho_rx
    csi = trainSet.csi
    if training_flag is False:
        ts = 0

    H_ = np.zeros((ts, 2 * Mr, 2 * Nt))
    x_ = np.zeros((ts, 2 * Nt, 1))
    y_ = np.zeros((ts, 2 * Mr, 1))
    sigma2_ = np.zeros((ts, 1, 1))
    indicator_ = np.zeros((ts, 2**(mu//2), 2 * Nt), dtype=int)
    rw_inv_ = np.zeros((ts, 2 * Mr, 2 * Mr))
    # generate development samples:
    Hval_ = np.zeros((vs, 2 * Mr, 2 * Nt))
    xval_ = np.zeros((vs, 2 * Nt, 1))
    yval_ = np.zeros((vs, 2 * Mr, 1))
    sigma2val_ = np.zeros((vs, 1, 1))
    indicatorval_ = np.zeros((vs, 2 ** (mu // 2), 2 * Nt), dtype=int)
    rw_invval_ = np.zeros((vs, 2 * Mr, 2 * Mr))

    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    if channel_type == 'nr':
        rspat = nr_corr_channel(Mr, Nt, 'meda', a=0.0)
        rherm = cholesky(rspat).T
    snr_min, snr_max = 10, 25
    if csi == 2:
        Np = Nt
        if trainSet.wlmmse is None:
            if trainSet.snr == 'varying_':
                wlmmse, xp, rw_inv = [], [], []
                for i in range(snr_min, snr_max + 1):
                    wlmmsei, xpi, rwi = channel_est(trainSet, snr=i, test=False, orth=trainSet.orth)
                    wlmmse.append(wlmmsei)
                    xp.append(xpi)
                    rw_inv.append(None)
            else:
                wlmmse, xp, rw = channel_est(trainSet, snr=SNR, test=False, orth=trainSet.orth)
                rw_inv = None
            trainSet.wlmmse, trainSet.xp, trainSet.rw_inv_set = wlmmse, xp, rw_inv
    for i in range(ts + vs):
        if trainSet.snr == 'varying_':
            SNR = np.random.randint(low=snr_min, high=snr_max + 1)
        # generate bits and modulate
        bits = np.random.binomial(n=1, p=0.5, size=(Nt * mu,))  # label
        bits_mod = QAM_Modulation(bits, mu)
        x = bits_mod.reshape(Nt, 1)
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

        if csi == 2:
            noise = np.sqrt(sigma2 / 2) * (np.random.randn(Mr, Np) + 1j * np.random.randn(Mr, Np))
            if trainSet.snr == 'varying_':
                wlmmse = trainSet.wlmmse[SNR - snr_min].copy()
                xp = trainSet.xp[SNR - snr_min].copy()
                rw_inv = trainSet.rw_inv_set[SNR - snr_min]
            else:
                wlmmse, xp = trainSet.wlmmse, trainSet.xp
                rw_inv = trainSet.rw_inv_set
            if i < ts:
                rw_inv_[i] = rw_inv
            else:
                rw_invval_[i - ts] = rw_inv
            yp = H @ xp + noise
            yp_vec = yp.reshape(-1, 1)
            h_hat = wlmmse @ yp_vec
            H = h_hat.reshape(Mr, Nt)

        # convert complex into real
        x = np.concatenate((np.real(x), np.imag(x)))
        H = np.concatenate((np.concatenate((np.real(H), -np.imag(H)), axis=1),
                            np.concatenate((np.imag(H), np.real(H)), axis=1)))
        y = np.concatenate((np.real(y), np.imag(y)))

        # stack
        if i < ts:
            H_[i] = H
            x_[i] = x
            y_[i] = y
            indicator_[i] = indicator(bits, mu)
            sigma2_[i] = sigma2
        else:
            Hval_[(i - ts)] = H
            xval_[(i - ts)] = x
            yval_[(i - ts)] = y
            sigma2val_[i - ts] = sigma2
            indicatorval_[i - ts] = indicator(bits, mu)

    return y_, x_, H_, sigma2_, indicator_, rw_inv_, yval_, xval_, Hval_, sigma2val_, indicatorval_, rw_invval_


def corr_channel(Mr, Nt, rho_tx=0.5, rho_rx=0.5):
    Rtx_vec = np.ones(Nt)
    for i in range(1, Nt):
        Rtx_vec[i] = rho_tx ** i
    Rtx = toeplitz(Rtx_vec)
    if Mr == Nt and rho_tx == rho_rx:
        Rrx = Rtx
    else:
        Rrx_vec = np.ones(Mr)
        for i in range(1, Mr):
            Rrx_vec[i] = rho_rx ** i
        Rrx = toeplitz(Rrx_vec)

    # another way of constructing kronecker model
    # C = cholesky(np.kron(Rtx,Rrx))    # complex correlation
    # C = sqrtm(np.sqrt(np.kron(Rtx, Rrx)))  # power field correlation--what's an equivalent model?
    # return C

    sqrtRtx = sqrtm(Rtx)  # sqrt decomposition for power field

    if Mr == Nt and rho_tx == rho_rx:
        sqrtRrx = sqrtRtx
    else:
        sqrtRrx = sqrtm(Rrx)

    return sqrtRtx, sqrtRrx


def detector(sys, H, x, y, sigma2, MSE, model=None):
    detect_type = sys.detect_type
    if sys.use_OFDM:
        Mr, Nt = sys.Mr * sys.K, sys.Nt * sys.K
    else:
        Mr, Nt = sys.Mr, sys.Nt
    mu = sys.mu
    T = sys.T
    if detect_type == 'ZF':  # ZF
        HT = H.T
        x_hat = la.inv(HT @ H) @ HT @ y
    elif detect_type == 'MMSE':  # MMSE
        HT = H.T
        x_hat = la.inv(HT @ H + sigma2 / 2 * np.eye(2 * Nt)) @ HT @ y
    elif detect_type == 'EP_real_v3':
        x_hat, mse = EP_real_v3(x, H, y, sigma2 / 2, T=T, mu=mu, modified=sys.modified, rw_inv=sys.rw_inv)
        MSE += mse
    elif detect_type == 'GEPNet':
        x = x.reshape((1, 2 * Nt, 1))
        y = y.reshape((1, 2 * Mr, 1))
        H = H.reshape((1, 2 * Mr, 2 * Nt))
        _, x_hat, _ = model(x, y, H, sigma2)
        mse = np.mean(abs(x_hat - x) ** 2)
        MSE += np.array([mse])
        x_hat = x_hat.numpy().reshape(-1, 1)
    elif detect_type == 'ML':
        x_hat = mlSolver(y, H, mu).reshape(-1, 1)
        MSE += np.mean((x - x_hat) ** 2)
    else:
        raise RuntimeError('The selected detector does not exist!')

    return x_hat, MSE


def mlSolver(y, h_real, mu):
    # status = []
    m, n = h_real.shape[0], h_real.shape[1]
    model = Model('mimo')
    M = 2**(mu // 2)
    sigConst = np.linspace(-M + 1, M - 1, M)
    sigConst /= np.sqrt((sigConst ** 2).mean())
    sigConst /= np.sqrt(2.)  # Each complex transmitted signal will have two parts
    z = model.addVars(n, M, vtype=GRB.BINARY, name='z')
    s = model.addVars(n, ub=max(sigConst)+.1, lb=min(sigConst)-.1, name='s')
    e = model.addVars(m, ub=200.0, lb=-200.0, vtype=GRB.CONTINUOUS, name='e')
    model.update()

    ### Constraints and variables definitions
    # define s[i]
    for i in range(n):
        model.addConstr(s[i] == quicksum(z[i,j]*sigConst[j] for j in range(M)))
    # constraint on z[i,j]
    model.addConstrs((z.sum(j,'*')==1 for j in range(n)), name='const1')
    # define e
    for i in range(m):
        e[i] = quicksum(h_real[i,j] * s[j] for j in range(n)) - y[i]

    ### define the objective function
    obj = e.prod(e)
    model.setObjective(obj, GRB.MINIMIZE)
    model.Params.logToConsole = 0
    model.setParam('TimeLimit', 100)
    model.update()

    model.optimize()

    # retrieve optimization result
    solution = model.getAttr('X', s)
    # status.append(model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL)
    # print(GRB.OPTIMAL, model.getAttr(GRB.Attr.Status))
    if model.getAttr(GRB.Attr.Status) == 9:
        print(np.linalg.cond(h_real))
    x_hat = []
    for num in solution:
        x_hat.append(solution[num])
    return np.array(x_hat)


def nr_corr_channel(Mr, Nt, corr_level, a):
    if corr_level == 'low':
        alpha, beta = 0, 0
    elif corr_level == 'med':
        alpha, beta = 0.3, 0.9
    elif corr_level == 'meda':
        alpha, beta = 0.3, 0.3874
    else:
        alpha, beta = 0.9, 0.9

    # Generate correlation matrix of NodeB side
    if Mr == 1:
        renb = 1
    elif Mr == 2:
        renb = toeplitz([1, alpha])
    elif Mr == 4:
        renb = toeplitz([1, alpha ** (1 / 9), alpha ** (4 / 9), alpha])
    elif Mr == 8:
        renb = toeplitz([1, alpha ** (1 / 49), alpha ** (4 / 49), alpha ** (9 / 49),
                         alpha ** (16 / 49), alpha ** (25 / 49), alpha ** (36 / 49), alpha])
    else:
        renb = np.eye(Mr)

    # Generate correlation matrix of UE side
    if Nt == 1:
        rue = 1
    elif Nt == 2:
        rue = toeplitz([1, beta])
    elif Nt == 4:
        rue = toeplitz([1, beta ** (1 / 9), beta ** (4 / 9), beta])
    elif Nt == 8:
        rue = toeplitz([1, beta ** (1 / 49), beta ** (4 / 49), beta ** (9 / 49),
                        beta ** (16 / 49), beta ** (25 / 49), beta ** (36 / 49), beta])
    else:
        rue = np.eye(Nt)

    # combined spatial correlation matrix
    rspat = np.kron(renb, rue)

    # "a" is a scaling factor such that the smallest value is used to make Rspat a positive semi-definite
    rspat = (rspat + a * np.eye(Mr*Nt)) / (1 + a)

    return rspat


def channel_est(sysin, snr, test=True, orth=True):
    if test:
        Mr, Nt, mu = sysin.Mr, sysin.Nt, sysin.mu
    else:
        Mr, Nt, mu = sysin.m, sysin.n, sysin.mu
    channel_type, rho_tx, rho_rx = sysin.channel_type, sysin.rho_tx, sysin.rho_rx
    Np = sysin.Np  # the number of pilot vectors
    if channel_type == 'corr':
        sqrtRtx, sqrtRrx = corr_channel(Mr, Nt, rho_tx=rho_tx, rho_rx=rho_rx)
    print('calculate covariance and LMMSE weight matrix for CE')
    num = 10000
    rhh = np.zeros((Mr * Nt, Mr * Nt), dtype=complex)
    for n in range(num):
        H = np.sqrt(1 / 2 / Mr) * (np.random.randn(Mr, Nt)
                                   + 1j * np.random.randn(Mr, Nt))  # Rayleigh MIMO channel
        if channel_type == 'corr':  # Correlated MIMO channel
            H = sqrtRrx @ H @ sqrtRtx
        h = H.reshape(-1, 1)
        rhh += h @ np.conj(h.T)
    rhh /= num
    if orth:
        xp = dft(Np)[:Nt, :]  # orthogonal pilots (nt, np)
    else:
        xp = 1 / np.sqrt(2) * (2 * np.random.binomial(1, 0.5, size=(Nt, Np)) - 1 +
                               1j * (2 * np.random.binomial(1, 0.5, size=(Nt, Np)) - 1))  # QPSK pilots
    sigma2 = 10 ** (-snr / 10)
    yp = np.zeros((Mr, Np), dtype=complex)
    wlmmse = lmmse_ce(xp, yp, sigma2, rhh)
    print('end')
    if sysin.modified:
        re = rhh - wlmmse @ np.kron(np.eye(Mr), xp.T) @ rhh
        rE = 0
        for i in range(Nt):
            rE += re[i * Mr:(i + 1) * Mr, i * Mr:(i + 1) * Mr]
        rw = 1 * rE + sigma2 * np.eye(Mr)
        rw = np.concatenate((np.concatenate((np.real(rw / 2), np.zeros((Mr, Mr))), axis=1),
                            np.concatenate((np.zeros((Mr, Mr)), np.real(rw / 2)), axis=1)))
    else:
        rw = None
    return wlmmse, xp, rw
