#!/usr/bin/python
import numpy as np

SNR_train = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def display_trained_vars(load_filename, save_filename, seq):
    with open(save_filename, 'a') as f:
        try:
            # f.write('\nSNR='+str(SNR_train[seq])+'\n')
            f.write('\n' + load_filename.replace(directory, '') + '\n')
            for k, d in np.load(load_filename).items():
                if k == 'log':
                    f.write(str(k))
                    f.write(str(d) + '\n')
            print('success')
        except IOError:
            pass
        f.close()


Mr = 4
Nt = 4
mu = 4
SNR = 16
loss = 'final'
coding = False
T = 10
directory = './'+ ('coded/' if coding else '') + 'GEPNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB'
load_filename = directory + '/' + 'GEPNet_' + str(Mr) + 'x' + str(Nt) + '_' + str(2 ** mu) + 'QAM_' + str(SNR) + 'dB_' + \
                loss + '_T' + str(T) + ('_coding' if coding else '') + '_paper_damping02.npz'
save_filename = directory + '/' + 'log.txt'

display_trained_vars(load_filename, save_filename, 0)

# for i in range(8):
# 	display_trained_vars('OAMP2_bg_giid_QPSK32MIMO'+str(i)+'.npz','QPSK_32MIMO_2_tsp.txt',i)
