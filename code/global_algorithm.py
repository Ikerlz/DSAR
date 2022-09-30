#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/8/10 01:15 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : sar_algo.py
# @desc :
import time

import numpy as np
import scipy.sparse as sp
import copy
# from util0 import *
# from tqdm import tqdm
from sklearn import random_projection
# from scipy.io import loadmat
# import h5py


def sar_algorithm(beta_init, rho_init, data_iter):
    p = len(beta_init)
    # data_iter is a N * (2p + 5) matrix.

    omega = data_iter[:, 0]  # here omega is diag(W^T W)
    W_Y = data_iter[:, 1]  # W_{i.}Y
    Wt_Y = data_iter[:, 2]  # W^T_{i.}Y
    Wt_W_Y = data_iter[:, 3]  # W^T_{i.}WY
    Wt_X = data_iter[:, 4:(p + 4)]  # W^T_{i.}X
    X = data_iter[:, (p + 4):(2 * p + 4)]  # X
    Y = data_iter[:, (2 * p + 4)]  # Y

    Nk, p = X.shape

    rho = copy.deepcopy(rho_init)
    beta = copy.deepcopy(beta_init)
    Qk_diff = 1
    Qk_old = 0
    max_iter = 20
    iter_num = 0
    start = time.time()
    while (Qk_diff > 1e-3) & (iter_num < max_iter):
        temp = 1 / (1 + rho ** 2 * omega)
        # D_k = np.diag(temp)  # Nk by Nk
        D_k = sp.diags(temp)
        # D_k_dev1 = - np.diag((2 * rho) * omega * temp ** 2)  # Nk by Nk
        D_k_dev1 = - sp.diags((2 * rho) * omega * temp ** 2)  # Nk by Nk
        # D_k_dev2 = np.diag(8 * rho ** 2 * omega ** 2 * temp ** 3 - 2 * omega * temp ** 2)  # Nk by Nk
        D_k_dev2 = sp.diags(8 * rho ** 2 * omega ** 2 * temp ** 3 - 2 * omega * temp ** 2)  # Nk by Nk
        G1 = Y - rho * W_Y - rho * Wt_Y + rho ** 2 * Wt_W_Y  # Nk by 1
        G1_dev1_rho = -W_Y - Wt_Y + 2 * rho * Wt_W_Y  # Nk by 1
        G1_dev2_rho = 2 * Wt_W_Y  # Nk by 1
        G2 = X - rho * Wt_X  # Nk by p
        G2_dev1_rho = -Wt_X  # Nk by p
        temp1 = G1 - G2 @ beta  # Nk by 1
        temp1_dev1_rho = G1_dev1_rho - G2_dev1_rho @ beta  # Nk by 1
        Fk = D_k @ temp1  # Nk by 1
        Fk_dev1_rho = D_k_dev1 @ temp1 + D_k @ temp1_dev1_rho  # Nk by 1
        Fk_dev1_beta = - D_k @ G2  # Nk by p
        Fk_dev1 = np.hstack((Fk_dev1_rho.reshape((-1, 1)), Fk_dev1_beta))  # Nk by (p+1)
        Fk_dev2_rho = D_k_dev2 @ temp1 + 2 * D_k_dev1 @ temp1_dev1_rho + D_k @ G1_dev2_rho  # Nk by 1
        Fk_dev1_rho_beta = - D_k_dev1 @ G2 - D_k @ G2_dev1_rho  # Nk by p

        Qk = Fk @ Fk
        Qk_dev1 = 2 * Fk_dev1.T @ Fk  # p+1 dim
        temp2 = np.zeros((p + 1, p + 1))
        temp2_rho = 2 * Fk @ Fk_dev2_rho
        temp2_rho_beta = 2 * Fk @ Fk_dev1_rho_beta
        temp2_theta = np.append(temp2_rho, temp2_rho_beta)
        temp2[:, 0] = temp2_theta
        temp2[0, :] = temp2_theta
        Qk_dev2 = 2 * Fk_dev1.T @ Fk_dev1 + 2 * temp2

        gradient = np.linalg.inv(Qk_dev2) @ Qk_dev1

        if np.abs(gradient[0]) > 5:
            gradient[0] = gradient[0] * 0.01
        if np.max(np.abs(gradient[1:])) > 5:
            gradient[1:] = 0.01 * gradient[1:]
        rho = rho - gradient[0]
        beta = beta - gradient[1:]
        Qk_diff = np.abs(Qk - Qk_old)
        Qk_old = Qk
        iter_num += 1
    end = time.time()

    return np.append(rho, beta), np.round(end - start, 4)


if __name__ == '__main__':
    # TODO: For real case
    data = sp.load_npz("/root/All_estimate_data.npz").toarray()
    global_t = 0
    for _ in range(10):
        res, tt = sar_algorithm(beta_init=np.zeros(5), rho_init=0, data_iter=data)
        global_t += tt / 10
    print(global_t)
    # N_list =  [5000 * i for i in range(1, 11)]
    # for N in N_list:
    #     ave_time = 0
    #     for i in tqdm(range(1, 11)):
    #         data = h5py.File("./dist_data_0918/N{}_{}_SBM.mat".format(N, i))["res"].value.T
    #         res, tt = sar_algorithm(beta_init=np.zeros(5), rho_init=0, data_iter=data)
    #         ave_time += tt / 10
    #     print("N={};Time={}".format(N, ave_time))