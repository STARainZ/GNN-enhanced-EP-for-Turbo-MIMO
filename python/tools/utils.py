#!/usr/bin/python
from __future__ import division
import numpy as np
import math
import os
import time
import numpy.linalg as la

sqrt = np.sqrt
pi = math.pi

_QPSK_mapping_table = {
    (0,1): (-1+1j,), (1,1): (1+1j,),
    (0,0): (-1-1j,), (1,0): (1-1j,)
}

_QPSK_demapping_table = {v: k for k, v in _QPSK_mapping_table.items()}

_QPSK_Constellation = np.array([[-1-1j], [-1+1j],
                                [1-1j], [1+1j]])

_16QAM_mapping_table = {
    (0,0,1,0): (-3+3j,), (0,1,1,0): (-1+3j,), (1,1,1,0): (1+3j,), (1,0,1,0): (3+3j,),
    (0,0,1,1): (-3+1j,), (0,1,1,1): (-1+1j,), (1,1,1,1): (1+1j,), (1,0,1,1): (3+1j,),
    (0,0,0,1): (-3-1j,), (0,1,0,1): (-1-1j,), (1,1,0,1): (1-1j,), (1,0,0,1): (3-1j,),
    (0,0,0,0): (-3-3j,), (0,1,0,0): (-1-3j,), (1,1,0,0): (1-3j,), (1,0,0,0): (3-3j,)
}

_16QAM_demapping_table = {v: k for k, v in _16QAM_mapping_table.items()}

_16QAM_Constellation = np.array([-3-3j,-3-1j,-3+3j,-3+1j,
                                -1-3j,-1-1j,-1+3j,-1+1j,
                                +3-3j,+3-1j,+3+3j,+3+1j,
                                +1-3j,+1-1j,+1+3j,+1+1j]).reshape(-1,1)

_64QAM_mapping_table = {
    (0,0,0,1,0,0): (-7+7j,), (0,0,1,1,0,0): (-5+7j,), (0,1,1,1,0,0): (-3+7j,), (0,1,0,1,0,0): (-1+7j,), (1,1,0,1,0,0): (1+7j,), (1,1,1,1,0,0): (3+7j,), (1,0,1,1,0,0): (5+7j,), (1,0,0,1,0,0): (7+7j,),
    (0,0,0,1,0,1): (-7+5j,), (0,0,1,1,0,1): (-5+5j,), (0,1,1,1,0,1): (-3+5j,), (0,1,0,1,0,1): (-1+5j,), (1,1,0,1,0,1): (1+5j,), (1,1,1,1,0,1): (3+5j,), (1,0,1,1,0,1): (5+5j,), (1,0,0,1,0,1): (7+5j,),
    (0,0,0,1,1,1): (-7+3j,), (0,0,1,1,1,1): (-5+3j,), (0,1,1,1,1,1): (-3+3j,), (0,1,0,1,1,1): (-1+3j,), (1,1,0,1,1,1): (1+3j,), (1,1,1,1,1,1): (3+3j,), (1,0,1,1,1,1): (5+3j,), (1,0,0,1,1,1): (7+3j,),
    (0,0,0,1,1,0): (-7+1j,), (0,0,1,1,1,0): (-5+1j,), (0,1,1,1,1,0): (-3+1j,), (0,1,0,1,1,0): (-1+1j,), (1,1,0,1,1,0): (1+1j,), (1,1,1,1,1,0): (3+1j,), (1,0,1,1,1,0): (5+1j,), (1,0,0,1,1,0): (7+1j,),
    (0,0,0,0,1,0): (-7-1j,), (0,0,1,0,1,0): (-5-1j,), (0,1,1,0,1,0): (-3-1j,), (0,1,0,0,1,0): (-1-1j,), (1,1,0,0,1,0): (1-1j,), (1,1,1,0,1,0): (3-1j,), (1,0,1,0,1,0): (5-1j,), (1,0,0,0,1,0): (7-1j,),
    (0,0,0,0,1,1): (-7-3j,), (0,0,1,0,1,1): (-5-3j,), (0,1,1,0,1,1): (-3-3j,), (0,1,0,0,1,1): (-1-3j,), (1,1,0,0,1,1): (1-3j,), (1,1,1,0,1,1): (3-3j,), (1,0,1,0,1,1): (5-3j,), (1,0,0,0,1,1): (7-3j,),
    (0,0,0,0,0,1): (-7-5j,), (0,0,1,0,0,1): (-5-5j,), (0,1,1,0,0,1): (-3-5j,), (0,1,0,0,0,1): (-1-5j,), (1,1,0,0,0,1): (1-5j,), (1,1,1,0,0,1): (3-5j,), (1,0,1,0,0,1): (5-5j,), (1,0,0,0,0,1): (7-5j,),
    (0,0,0,0,0,0): (-7-7j,), (0,0,1,0,0,0): (-5-7j,), (0,1,1,0,0,0): (-3-7j,), (0,1,0,0,0,0): (-1-7j,), (1,1,0,0,0,0): (1-7j,), (1,1,1,0,0,0): (3-7j,), (1,0,1,0,0,0): (5-7j,), (1,0,0,0,0,0): (7-7j,)
}

_64QAM_demapping_table = {v: k for k, v in _64QAM_mapping_table.items()}

_64QAM_Constellation = np.array([ -7-7j,-7-5j,-7-1j,-7-3j,-7+7j,-7+5j,-7+1j,-7+3j,
                              -5-7j,-5-5j,-5-1j,-5-3j,-5+7j,-5+5j,-5+1j,-5+3j,
                              -1-7j,-1-5j,-1-1j,-1-3j,-1+7j,-1+5j,-1+1j,-1+3j,
                              -3-7j,-3-5j,-3-1j,-3-3j,-3+7j,-3+5j,-3+1j,-3+3j,
                              +7-7j,+7-5j,+7-1j,+7-3j,+7+7j,+7+5j,+7+1j,+7+3j,
                              +5-7j,+5-5j,+5-1j,+5-3j,+5+7j,+5+5j,+5+1j,+5+3j,
                              +1-7j,+1-5j,+1-1j,+1-3j,+1+7j,+1+5j,+1+1j,+1+3j,
                              +3-7j,+3-5j,+3-1j,+3-3j,+3+7j,+3+5j,+3+1j,+3+3j]).reshape(-1,1)

_QPSK_onehot_mapping_table = {  # -1 +1
    (0): (0, 1), (1): (1, 0)
}

_16QAM_onehot_mapping_table = {   # -3 -1 +1 +3
    (0, 0): (0, 0, 0, 1), (0, 1): (0, 0, 1, 0),
    (1, 1): (0, 1, 0, 0), (1, 0): (1, 0, 0, 0)
}

_64QAM_onehot_mapping_table = {  # -7 -5 -3 -1 +1 +3 +5 +7
    (0, 0, 0): (0, 0, 0, 0, 0, 0, 0, 1), (0, 0, 1): (0, 0, 0, 0, 0, 0, 1, 0),
    (0, 1, 1): (0, 0, 0, 0, 0, 1, 0, 0), (0, 1, 0): (0, 0, 0, 0, 1, 0, 0, 0),
    (1, 1, 0): (0, 0, 0, 1, 0, 0, 0, 0), (1, 1, 1): (0, 0, 1, 0, 0, 0, 0, 0),
    (1, 0, 1): (0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0): (1, 0, 0, 0, 0, 0, 0, 0)
}

sq2 = sqrt(2)
sq10 = sqrt(10)
sq42 = sqrt(42)


def indicator(bits, mu):
    indicator = []
    for i in range(int(len(bits) / (mu//2) )):
        if mu == 2:
            indicator.append(list(_QPSK_onehot_mapping_table.get(bits[i])))  # shape(-1, 2)
        elif mu == 4:
            indicator.append(list(_16QAM_onehot_mapping_table.get(tuple(bits[2*i:2*(i+1)]))))  # shape(-1, 4)
        else:
            indicator.append(list(_64QAM_onehot_mapping_table.get(tuple(bits[3 * i:3 * (i + 1)]))))  # shape(-1, 8)
    indicator = np.asarray(indicator).T  # shape(2/4/8, -1)
    indicator = np.concatenate((indicator[:,0::2], indicator[:, 1::2]), axis=1)  # (real, imag)
    return indicator


def QAM_Modulation(bits,mu):
    if mu == 1:
        bits_mod = (2*bits-1).reshape(int(len(bits)),1)
    elif mu == 2:
        bits_mod = Modulation(bits)/sq2
    elif mu == 4:
        bits_mod = Modulation_16(bits)/sq10
    else:
        bits_mod = Modulation_64(bits)/sq42
    return bits_mod


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/2), 2))  # real & imag
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)  # This is just for QAM modulation
#    return np.concatenate((2*bit_r[:,0]-1, 2*bit_r[:,1]-1))


# mapping
def Modulation_16(bits):
    bit_r = bits.reshape((int(len(bits)/4), 4))
    bit_mod = []
    for i in range(int(len(bits)/4)):
        bit_mod.append(list(_16QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def Modulation_64(bits):
    bit_r = bits.reshape((int(len(bits)/6), 6))
    bit_mod = []
    for i in range(int(len(bits)/6)):
        bit_mod.append(list(_64QAM_mapping_table.get(tuple(bit_r[i]))))
    return np.asarray(bit_mod).reshape((-1,))


def QAM_Demodulation(bits_mod,mu):
    if mu == 1:
        bits_demod = abs(bits_mod+1) >= abs(bits_mod-1)
        bits_demod = bits_demod.astype(np.int32).reshape(-1)
    elif mu == 2:
        bits_demod = Demodulation(bits_mod*sq2)
    elif mu == 4:
        bits_demod = Demodulation_16(bits_mod*sq10)
    else:
        bits_demod = Demodulation_64(bits_mod*sq42)
    return bits_demod


def Demodulation(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((4,1))
        min_distance_index = np.argmin(abs(tmp - _QPSK_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_QPSK_demapping_table[
            tuple(_QPSK_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_16(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((16,1))
        min_distance_index = np.argmin(abs(tmp - _16QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_16QAM_demapping_table[
            tuple(_16QAM_Constellation[min_distance_index])])))
    return X_pred


def Demodulation_64(bits_mod):
    X_pred = np.array([])
    for i in range(len(bits_mod)):
        tmp = bits_mod[i] * np.ones((64,1))
        min_distance_index = np.argmin(abs(tmp - _64QAM_Constellation))
        X_pred = np.concatenate((X_pred,np.array(_64QAM_demapping_table[
            tuple(_64QAM_Constellation[min_distance_index])])))
    return X_pred


def NLE(vle,ule,orth=True,mu=2,SE=False,x=None,EP=False,soft=False,norm=1):
    if soft:
        ext_probs = np.zeros((ule.shape[0], 2**(mu//2)))
    # for QPSK signal
    if mu == 2:  # {-1,+1}
        P0 = np.maximum(np.exp(-(-1/sq2/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq2/norm-ule)**2/(2*vle)),1e-100)
        u_post = (P1-P0) / (P1+P0)/sq2/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P0*(u_post+1/sq2/norm)**2+P1*(u_post-1/sq2/norm)**2)/(P1+P0)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P0.reshape(-1), P1.reshape(-1)
    elif mu == 4:  # {-3,-1,+1,+3}
        P_3 = np.maximum(np.exp(-(-3/sq10/norm-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq10/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq10/norm-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq10/norm-ule)**2/(2*vle)),1e-100)
        u_post = (-3*P_3-P_1+P1+3*P3) / (P_3+P_1+P1+P3)/sq10/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_3*(u_post+3/sq10/norm)**2+P_1*(u_post+1/sq10/norm)**2 +
                      P1*(u_post-1/sq10/norm)**2+P3*(u_post-3/sq10/norm)**2)/(P_3+P_1+P1+P3)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P_3.reshape(-1), P_1.reshape(-1)
            ext_probs[:, 2], ext_probs[:, 3] = P3.reshape(-1), P1.reshape(-1)
    else:  # {-7,-5,-3,-1,+1,+3,+5,+7}
        P_7 = np.maximum(np.exp(-(-7/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_5 = np.maximum(np.exp(-(-5/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_3 = np.maximum(np.exp(-(-3/sq42/norm-ule)**2/(2*vle)),1e-100)
        P_1 = np.maximum(np.exp(-(-1/sq42/norm-ule)**2/(2*vle)),1e-100)
        P1 = np.maximum(np.exp(-(1/sq42/norm-ule)**2/(2*vle)),1e-100)
        P3 = np.maximum(np.exp(-(3/sq42/norm-ule)**2/(2*vle)),1e-100)
        P5 = np.maximum(np.exp(-(5/sq42/norm-ule)**2/(2*vle)),1e-100)
        P7 = np.maximum(np.exp(-(7/sq42/norm-ule)**2/(2*vle)),1e-100)
        u_post = (-7*P_7-5*P_5-3*P_3-P_1+P1+3*P3+5*P5+7*P7) / \
                 (P_7+P_5+P_3+P_1+P1+P3+P5+P7)/sq42/norm
        if SE is True:
            v_post = np.mean((x-u_post)**2)
        else:
            v_post = (P_7*(u_post+7/sq42/norm)**2+P_5*(u_post+5/sq42/norm)**2 +
                      P_3*(u_post+3/sq42/norm)**2+P_1*(u_post+1/sq42/norm)**2 +
                      P1*(u_post-1/sq42/norm)**2+P3*(u_post-3/sq42/norm)**2 +
                      P5*(u_post-5/sq42/norm)**2+P7*(u_post-7/sq42/norm)**2) / \
                     (P_7+P_5+P_3+P_1+P1+P3+P5+P7)
        if soft:
            ext_probs[:, 0], ext_probs[:, 1] = P_7.reshape(-1), P_5.reshape(-1)
            ext_probs[:, 2], ext_probs[:, 3] = P_1.reshape(-1), P_3.reshape(-1)
            ext_probs[:, 4], ext_probs[:, 5] = P7.reshape(-1), P5.reshape(-1)
            ext_probs[:, 6], ext_probs[:, 7] = P1.reshape(-1), P3.reshape(-1)
    if EP is False:
        v_post = np.mean(v_post)

    if orth:
        u_orth = (u_post/v_post-ule/vle)/(1/v_post-1/vle)
        v_orth = 1/(1/v_post-1/vle)
    else:
        u_orth = u_post
        v_orth = v_post

    if soft:
        return u_post,v_post,u_orth,v_orth,ext_probs
    return u_post,v_post,u_orth,v_orth


def de2bi(decimal, order):  # decimal to binary
    binary = np.zeros((len(decimal), order), dtype = int)
    for i in range(len(decimal)):
        temp = bin(decimal[i])[2:]  # remove '0b'
        for j in range(order - len(temp)):
            binary[i, j] = 0
        for j in range(len(temp)):
            binary[i, order - len(temp) + j] = temp[j]
    return binary


def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


def lmmse_ce(xp_mat, yp_mat, sigma2, rhh):
    Nr, Np = yp_mat.shape
    A = np.kron(np.eye(Nr), xp_mat.T)
    AH = np.conj(A.T)
    wlmmse = rhh @ AH @ la.inv(A @ rhh @ AH + sigma2 * np.eye(Nr * Np))
    return wlmmse