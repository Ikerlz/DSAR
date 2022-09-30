#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/9/30 10:38 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : test111111.py
# @desc :
import time
import copy

from pyspark import SparkContext
from pyspark import SparkConf
import sys
print(sys.executable)
# import findspark
import numpy as np
# from util0 import *
import scipy.sparse as sp
from sklearn import random_projection
import os
os.environ['PYSPARK_PYTHON'] = './pyspark_env/pyspark_env/bin/python'



class SparseDistributedSAR(object):
    def __init__(self, mode="simulate", extra_params=None):
        """
        :param mode: simulate or real
        :param extra_params:
            if mode = simulate:
                extra_params = {
                    "net_model":...
                    "data_num":...
                    "worker_num":...
                    "group_num":...  just for SBM
                    "rho":...
                    "beta":...
                    "rho_init":...
                    "beta_init":...
                    "seed1":...
                    "seed2":...
                    "spark_context":...
                    "method":...
                }
        """
        if mode == "simulate":
            # read the variables from the extra_params
            self.net_model = extra_params["net_model"]
            self.data_num = extra_params["data_num"]
            self.worker_num = extra_params["worker_num"]
            self.rho = extra_params["rho"]
            self.beta = extra_params["beta"]
            self.rho_init = extra_params["rho_init"]
            self.beta_init = extra_params["beta_init"]
            self.group_num = extra_params["group_num"]
            self.seed = extra_params["seed"]
            self.seed1 = extra_params["seed1"]
            self.seed2 = extra_params["seed2"]
            self.proj_const = extra_params["proj_const"]
            self.method = extra_params["method"]
            self.sigma_X = extra_params["sigma_X"]
            self.sigma_eps = extra_params["sigma_eps"]
            # random projection
            r = int(np.ceil(self.proj_const * np.log(self.data_num)))
            transformer1 = random_projection.GaussianRandomProjection(random_state=self.seed1)
            transformer2 = random_projection.GaussianRandomProjection(random_state=self.seed2)
            self.R1 = sp.csc_matrix(transformer1._make_random_matrix(n_components=r, n_features=self.data_num))
            self.R2 = sp.csc_matrix(transformer2._make_random_matrix(n_components=r, n_features=self.data_num))
            self.X, self.Y, self.W = self.generate_simulate_data()
            self.omega_vec = np.diag(self.W.T @ self.W)
            self.onestep_rho = None
            self.onestep_beta = None
            self.twostep_rho = None
            self.twostep_beta = None
            self.onestep_se = None
            self.twostep_se = None

        elif mode == "real":
            # read the variables from the extra_params
            self.data_num = extra_params["data_num"]
            self.worker_num = extra_params["worker_num"]
            self.rho_init = extra_params["rho_init"]
            self.beta_init = extra_params["beta_init"]
            # self.seed = extra_params["seed"]
            self.seed1 = extra_params["seed1"]
            self.seed2 = extra_params["seed2"]
            self.proj_const = extra_params["proj_const"]
            self.method = extra_params["method"]
            self.sigma_X = extra_params["sigma_X"]
            self.sigma_eps = extra_params["sigma_eps"]
            self.omega_vec = extra_params["omega_vec"]
            # random projection
            self.r = int(np.ceil(self.proj_const * np.log(self.data_num)))
            transformer1 = random_projection.SparseRandomProjection(random_state=self.seed1)
            transformer2 = random_projection.SparseRandomProjection(random_state=self.seed2)
            # transformer1 = random_projection.GaussianRandomProjection(random_state=self.seed1)
            # transformer2 = random_projection.GaussianRandomProjection(random_state=self.seed2)
            self.R1 = transformer1._make_random_matrix(n_components=self.r, n_features=self.data_num)
            self.R2 = transformer2._make_random_matrix(n_components=self.r, n_features=self.data_num)
            # self.R1 = sp.csc_matrix(transformer1._make_random_matrix(n_components=r, n_features=self.data_num))
            # self.R2 = sp.csc_matrix(transformer2._make_random_matrix(n_components=r, n_features=self.data_num))
            self.oneshot_rho = None
            self.oneshot_beta = None
            self.onestep_rho = None
            self.onestep_beta = None
            self.twostep_rho = None
            self.twostep_beta = None
        else:
            raise ValueError("The mode argument must be simulate or real, but {} is given".format(mode))



    def cal_Xi(self, W_k, Wt_k, Wt_W_k, index_k, rho):
        # W_k, Wt_k, Wt_W_k: Nk by N
        # omega_vec: Nk by 1
        omega_vec_k = self.omega_vec[index_k] # Nk by 1
        N = W_k.shape[1]
        I_k = sp.csc_matrix(sp.eye(N))[index_k, :] # Nk by N
        D_k = sp.csc_matrix(1 / (rho ** 2 * omega_vec_k.toarray() + 1)) # Nk by 1
        # D_k = rho ** 2 * omega_vec_k
        # D_k.data = 1 / (D_k.data + 1)
        # D_k_vec = np.zeros(N)
        # D_k_vec[index_k] = D_k # N by 1
        D_k_vec = sp.csc_matrix((D_k.data, index_k, [0, len(index_k)]), shape=(N, 1)) # N by 1
        D_k_dev1 = sp.csc_matrix(-2 * rho * omega_vec_k.toarray() * D_k.toarray()) # Nk by 1
        # D_k_dev1 = -2 * rho * omega_vec_k * D_k ** 2  # Nk by 1
        # tmp1 = sp.csc_matrix((rho ** 2 * Wt_W_k.data, Wt_W_k.indices, Wt_W_k.indptr), shape=Wt_W_k.shape)
        Xi_part1 = (I_k + rho * rho * Wt_W_k - rho * W_k - rho * Wt_k).T * D_k_dev1  # N by 1
        Xi_part2 = (W_k - rho * Wt_W_k).T * D_k # N by 1
        Xi_part3 = (Wt_k - rho * Wt_W_k).T * D_k # N by 1

        # Xi = (Xi_part1 - Xi_part2 - Xi_part3) * D_k_vec.T # N by N
        Xi_part4 = Xi_part1 - Xi_part2 - Xi_part3
        R1_Xi = self.R1 * Xi_part4
        Xi_R2 = D_k_vec.T * self.R2.T
        R2_Xi = self.R2 * Xi_part4
        Xi_R1 = D_k_vec.T * self.R1.T
        return R1_Xi * Xi_R2, R2_Xi * Xi_R1

        # return
        # return self.R1 * Xi * self.R2.T, self.R2 * Xi * self.R1.T

    def cal_V1(self, W_k, index_k, rho):
        # N = W_k.shape[1]
        # self.omega_vec: N by 1
        D = sp.diags(1 / (1 + rho ** 2 * self.omega_vec.toarray().T[0])).tocsc() # N by N
        # V1 = sp.lil_matrix((N, N)) # N by N
        # V1[:, index_k] = (D[:, index_k] - rho * D * W_k.T).tocsc() # N by N
        # return
        return self.R1 * (D[:, index_k] - rho * D * W_k.T) * self.R2.T[index_k, :]

    def cal_V2(self, W_k, Wt_k, Wt_W_k, index_k, rho):
        N = W_k.shape[1]
        I_k = sp.eye(N).tocsc()[index_k, :] # Nk by N
        D = sp.diags(1 / (1 + rho ** 2 * self.omega_vec.toarray().T[0])).tocsc() # N by N
        D_dev1 = sp.diags(
            (-2 * rho * self.omega_vec.toarray().T[0]) / ((1 + rho ** 2 * self.omega_vec.toarray().T[0]) ** 2)
        ).tocsc() # N by N
        D_k_vec = sp.csc_matrix(D.diagonal()[index_k]).T # Nk by 1
        D_k_dev1_vec = sp.csc_matrix(D_dev1.diagonal()[index_k]).T # Nk by 1
        M_part1 = D_dev1 * (I_k + rho ** 2 * Wt_W_k - rho * W_k - rho * Wt_k).T * D_k_dev1_vec # N by 1
        M_part2 = D_dev1 * (Wt_k - rho * Wt_W_k).T * D_k_vec # N by 1
        M_part3 = D_dev1 * (W_k - rho * Wt_W_k).T * D_k_vec # N by 1
        M_part4 = D * (Wt_k - rho * Wt_W_k).T * D_k_dev1_vec # N by 1
        M_part5 = D * (W_k - rho * Wt_W_k).T * D_k_dev1_vec # N by 1
        M_part6 = D * Wt_W_k.T * D_k_vec # N by 1
        # V2 = (M_part1 + M_part6 - (M_part2 + M_part3 + M_part4 + M_part5)) * \
        #      (sp.csc_matrix(D.data).T - rho * Wt_k.T @ D_k_vec).T # N by N
        V2_part1 = M_part1 + M_part6 - (M_part2 + M_part3 + M_part4 + M_part5)
        V2_part2 = (sp.csc_matrix(D.data).T - rho * Wt_k.T @ D_k_vec).T
        R1_V2 = self.R1 * V2_part1
        V2_R2 = V2_part2 * self.R2.T

        return R1_V2 * V2_R2

        # return
        # return self.R1 * V2 * self.R2.T

    def cal_T1(self, W_k, WY_k, Wt_W_Y_k, index_k, rho):
        # N = W_k.shape[1]
        # I_k = np.eye(N)[index_k, :]
        D = sp.diags(1 / (1 + rho ** 2 * self.omega_vec.toarray().T[0])).tocsc() # N by N
        D_k_vec = D.diagonal()[index_k] # Nk by 1
        # D_k_vec = np.diag(D)[index_k]
        # T1 = ((WY_k - rho * Wt_W_Y_k) * D_k_vec) @ (D[index_k, :] - rho * W_k @ D)
        T1 = (sp.csc_matrix((WY_k - rho * Wt_W_Y_k) * D_k_vec)) * (D[index_k, :] - rho * W_k * D) # 1 by N

        # return
        return T1 * self.R1.T

    def cal_T2(self, W_k, Wt_Y_k, Wt_W_Y_k, index_k, rho):
        # N = W_k.shape[1]
        D = sp.diags(1 / (1 + rho ** 2 * self.omega_vec.toarray().T[0])).tocsc() # N by N
        # D_k_vec = np.diag(D)[index_k]
        D_k_vec = D.diagonal()[index_k]  # Nk by 1
        # temp1 = np.zeros(N)
        # temp1[index_k] = D_k_vec * (Wt_Y_k - rho * Wt_W_Y_k)
        T2 = (sp.csc_matrix((Wt_Y_k - rho * Wt_W_Y_k) * D_k_vec)) * (D[index_k, :] - rho * W_k * D)

        # return
        return T2 * self.R1.T

    def cal_T3(self, W_k, X_k, Wt_X_k, index_k, rho):
        D = sp.diags(1 / (1 + rho ** 2 * self.omega_vec.toarray().T[0])).tocsc() # N by N
        T3 = (X_k - rho * Wt_X_k).T * (D[index_k, :] - rho * W_k * D)

        # return
        return T3 * self.R1.T


    def _worker_estimate_algorithm(self, iterator):
        data_iter = [*iterator][0]
        # data_iter = sp.vstack([*iterator]).tocsc()
        # data_iter = np.array([*iterator])
        # p = len(self.beta_init)
        omega = data_iter[0].toarray().flatten()
        W_Y = data_iter[1].toarray().flatten()
        Wt_Y = data_iter[2].toarray().flatten()
        Wt_W_Y = data_iter[3].toarray().flatten()
        Wt_X = data_iter[4].toarray()
        X = data_iter[5].toarray()
        Y = data_iter[6].toarray().flatten()
        # data_iter is a N_k * (3N + 2p + 6) matrix.
        # omega = data_iter[:, 0].toarray().flatten()  # here omega is diag(W^T W)
        # W_Y = data_iter[:, 1].toarray().flatten()   # W_{i.}Y
        # Wt_Y = data_iter[:, 2].toarray().flatten()   # W^T_{i.}Y
        # Wt_W_Y = data_iter[:, 3].toarray().flatten()   # W^T_{i.}WY
        # Wt_X = data_iter[:, 4:(p + 4)].toarray()   # W^T_{i.}X
        # X = data_iter[:, (p + 4):(2 * p + 4)].toarray()   # X
        # Y = data_iter[:, (2 * p + 4)].toarray().flatten()   # Y

        Nk, p = X.shape

        if self.method == "onestep":
            rho = copy.deepcopy(self.rho_init)
            beta = copy.deepcopy(self.beta_init)
            Qk_diff = 1
            Qk_old = 0
            max_iter = 20
            iter_num = 0
            start = time.time()
            while (Qk_diff > 1e-3) & (iter_num < max_iter):
                temp = 1 / (1 + rho ** 2 * omega)
                D_k = sp.diags(temp)  # Nk by Nk
                D_k_dev1 = - sp.diags((2 * rho) * omega * temp ** 2)  # Nk by Nk
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
        elif self.method == "twostep":
            rho = self.onestep_rho
            beta = self.onestep_beta
            start = time.time()
            temp = 1 / (1 + rho ** 2 * omega)
            D_k = sp.diags(temp)  # Nk by Nk
            D_k_dev1 = - sp.diags(2 * rho * omega * temp ** 2)  # Nk by Nk
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
            Fk_dev2_rho_beta = - D_k_dev1 @ G2 - D_k @ G2_dev1_rho  # Nk by p

            Qk_dev1 = 2 * Fk_dev1.T @ Fk  # p+1 dim
            temp2 = np.zeros((p + 1, p + 1))
            temp2_rho = 2 * Fk @ Fk_dev2_rho
            temp2_rho_beta = 2 * Fk @ Fk_dev2_rho_beta
            temp2_theta = np.append(temp2_rho, temp2_rho_beta)
            temp2[:, 0] = temp2_theta
            temp2[0, :] = temp2_theta
            Qk_dev2 = 2 * Fk_dev1.T @ Fk_dev1 + temp2

            gradient = np.linalg.inv(Qk_dev2) @ Qk_dev1

            if np.abs(gradient[0]) > 5:
                gradient[0] = gradient[0] * 0.01
            if np.max(np.abs(gradient[1:])) > 5:
                gradient[1:] = 0.01 * gradient[1:]
            rho = rho - gradient[0]
            beta = beta - gradient[1:]
            end = time.time()
        else:
            raise ValueError("the method should be onestep or twostep, but {} is given.".format(self.method))

        return Nk, np.append(rho, beta), Qk_dev2, np.round(end - start, 4)

    def _worker_calculate_sigma1(self, iterator):
        # N = self.data_num
        data_iter = [*iterator][0]
        # omega = data_iter[0]
        W_Y = data_iter[1].toarray().flatten()
        Wt_Y = data_iter[2].toarray().flatten()
        Wt_W_Y = data_iter[3].toarray().flatten()
        Wt_X = data_iter[4]
        X = data_iter[5]
        # Y = data_iter[6]
        W = data_iter[7]
        Wt = data_iter[8]
        Wt_W = data_iter[9]
        index_list = data_iter[10].toarray().flatten().astype(int).tolist()
        # data_iter = sp.vstack([*iterator]).tocsc()
        # p = len(self.beta_init)
        # omega = data_iter[:, 0]  # here omega is diag(W^T W)
        # W_Y = data_iter[:, 1].toarray().flatten()  # W_{i.}Y
        # Wt_Y = data_iter[:, 2].toarray().flatten()  # W^T_{i.}Y
        # Wt_W_Y = data_iter[:, 3].toarray().flatten()  # W^T_{i.}WY
        # Wt_X = data_iter[:, 4:p + 4]  # W^T_{i.}X
        # X = data_iter[:, (p + 4):(2 * p + 4)]  # X
        # # Y = data_iter[:, (2 * p + 5)]  # Y
        # W = data_iter[:, (2 * p + 5): (2 * p + 5 + N)]  # W
        # Wt = data_iter[:, (2 * p + 5 + N): (2 * p + 5 + 2*N)]  # Wt
        # Wt_W = data_iter[:, (2 * p + 5 + 2*N): (2 * p + 5 + 3*N)]  # WtW
        # index_list = data_iter[:, -1].toarray().flatten().astype(int).tolist()  # index_list
        if self.method == "onestep":
            rho = self.onestep_rho
        elif self.method == "twostep":
            rho = self.twostep_rho
        else:
            raise ValueError("the method should be onestep or twostep, but {} is given.".format(self.method))
        R1_Xi_R2, R2_Xi_R1 = self.cal_Xi(W, Wt, Wt_W, index_list, rho)
        V1 = self.cal_V1(W, index_list, rho)
        V2 = self.cal_V2(W, Wt, Wt_W, index_list, rho)
        T1 = self.cal_T1(W, W_Y, Wt_W_Y, index_list, rho)
        T2 = self.cal_T2(W, Wt_Y, Wt_W_Y, index_list, rho)
        T3 = self.cal_T3(W, X, Wt_X, index_list, rho)
        return R1_Xi_R2, R2_Xi_R1, V1, V2, T1, T2, T3

    def _reduce_estimator(self, estimators):
        worker_df = np.array(estimators).reshape(-1, 4).T
        num_in_worker_list = worker_df[0].tolist()
        self.num_in_worker_list = num_in_worker_list
        # total_num = sum(num_in_worker_list)
        theta_list = worker_df[1].tolist()
        sigma_inv_list = worker_df[2].tolist()
        t = np.mean(worker_df[3].tolist())
        worker_num = len(theta_list)
        num_of_variable = len(theta_list[0])
        sigma_inv = np.zeros((num_of_variable, num_of_variable))
        sigma_inv_sum = np.zeros((num_of_variable, num_of_variable))
        sig_theta = np.zeros((num_of_variable, 1))
        reduce_start = time.time()
        for i in range(worker_num):
            # alpha = num_in_worker_list[i] / self.data_num
            sigma_k = sigma_inv_list[i]
            theta_k = theta_list[i].reshape(num_of_variable, 1)
            sigma_inv_sum = sigma_inv_sum + sigma_k
            sigma_inv = sigma_inv + sigma_k
            sig_theta = sig_theta + sigma_k @ theta_k
        if self.method == "onestep":
            theta_est = np.dot(np.linalg.inv(sigma_inv), sig_theta).reshape(-1, )
            reduce_end = time.time()
            self.onestep_rho = theta_est[0]
            self.onestep_beta = theta_est[1:]
            self.onestep_Sigma2 = sigma_inv_sum
            self.onestep_t = t + reduce_end - reduce_start
        elif self.method == "twostep":
            theta_est = np.dot(np.linalg.inv(sigma_inv), sig_theta).reshape(-1, )
            reduce_end = time.time()
            self.twostep_rho = theta_est[0]
            self.twostep_beta = theta_est[1:]
            self.twostep_Sigma2 = sigma_inv_sum
            self.twostep_t = self.onestep_t + t + reduce_end - reduce_start
        else:
            raise ValueError("the method should be onestep or twostep, but {} is given.".format(self.method))

    def _reduce_se(self, sigma1s, beta_vec):
        # R1_Xi_R2: R by R
        # R2_Xi_R1: R by R
        # V1: R by R
        # V2: R by R
        # T1: 1 by R
        # T2: 1 by R
        # T3: p by R
        worker_df = np.array(sigma1s).reshape(-1, 7).T
        # concatenate
        # p = len(beta_vec)
        # R1_Xi_R2_concat = sp.vstack(worker_df[0].tolist()) # KR by R
        # R2_Xi_R1_concat = sp.hstack(worker_df[1].tolist())  # R by KR
        # V1_concat = sp.hstack(worker_df[2].tolist()) # R by KR
        # V2_concat = sp.hstack(worker_df[3].tolist()) # R by KR
        # T1_concat = sp.hstack(worker_df[4].tolist()) # K by R
        # T2_concat = sp.hstack(worker_df[5].tolist()) # K by R
        # T3_concat = sp.hstack(worker_df[6].tolist())  # pK by R
        # block_slices = [slice(k*self.r, (k+1)*self.r) for k in range(self.worker_num)]
        # block_slices_p = [slice(k * p, (k + 1) * p) for k in range(self.worker_num)]

        # # calculation
        # sigma_tilde = beta_vec @ self.sigma_X @ beta_vec + self.sigma_eps
        #
        # Sigma1 = np.zeros((p + 1, p + 1))
        # p1_part1 = sum(np.array([[(R1_Xi_R2_concat * R2_Xi_R1_concat)[block_slices[i], block_slices[j]].diagonal().sum()
        #                for j in range(self.worker_num)] for i in range(self.worker_num)]))
        # p1_part2 = sum(np.array([[(V1_concat.T * V2_concat)[block_slices[i], block_slices[j]].diagonal().sum()
        #                for j in range(self.worker_num)] for i in range(self.worker_num)]))
        # p1_part3 = (T1_concat * T2_concat.T + T2_concat * T1_concat.T).sum() # K by K
        # p1_part4 = (T1_concat * T1_concat.T).sum() # K by K
        # Sigma1[0, 0] = 4 * (p1_part1 + p1_part2 + 1/sigma_tilde * p1_part3 + (1+(1-self.sigma_eps)/sigma_tilde) * p1_part4)
        #
        # p2 = -4 * np.array([((T1_concat * T3_concat.T).toarray().sum(axis=0))[[p*x+d for x in range(self.worker_num)]].sum() for d in range(p)])
        # Sigma1[0, 1:(p + 1)] = Sigma1[0, 1:(p + 1)] + p2.toarray()
        # Sigma1[1:(p + 1), 0] = Sigma1[1:(p + 1), 0] + p2.toarray()
        #
        # p3 = 4 * (T3_concat * T3_concat.T).toarray()
        # for k1 in range(self.worker_num):
        #     for k2 in range(self.worker_num):
        #         Sigma1[1:(p + 1), 1:(p + 1)] = Sigma1[1:(p + 1), 1:(p + 1)] +  p3[(p*k1): (p*k1), (p*k2): (p*k2)]



        R1_Xi_R2_list = worker_df[0].tolist()
        R2_Xi_R1_list = worker_df[1].tolist()
        V1_list = worker_df[2].tolist()
        V2_list = worker_df[3].tolist()
        T1_list = worker_df[4].tolist()
        T2_list = worker_df[5].tolist()
        T3_list = worker_df[6].tolist()
        sigma_tilde = beta_vec @ self.sigma_X @ beta_vec + self.sigma_eps
        p = len(beta_vec)
        Sigma1 = np.zeros((p+1, p+1))
        for k in range(self.worker_num):
            print(k)
            print("+"*50)
            for l in range(self.worker_num):
                # alpha = 1/np.sqrt(self.num_in_worker_list[k] * self.num_in_worker_list[l])
                # alpha = 1 / self.data_num
                p1 = 4 * (
                        np.sum((R1_Xi_R2_list[k] * R2_Xi_R1_list[l]).diagonal()) +
                        np.sum((V1_list[k].T * V2_list[l]).diagonal()) +
                        1/sigma_tilde * (T1_list[k] * T2_list[l].T + T2_list[l] * T1_list[l].T)[0, 0] +
                        (1+(1-self.sigma_eps)/sigma_tilde) * (T1_list[k] * T1_list[l].T)[0, 0]
                )
                p2 = -4 * T1_list[k] * T3_list[l].T
                p3 = 4 * T3_list[k] * T3_list[l].T
                Sigma1[0, 0] = Sigma1[0, 0] + p1
                Sigma1[0, 1:(p+1)] = Sigma1[0, 1:(p+1)] + p2.toarray()
                Sigma1[1:(p+1), 0] = Sigma1[1:(p+1), 0] + p2.toarray()
                Sigma1[1:(p+1), 1:(p+1)] = Sigma1[1:(p+1), 1:(p+1)] + p3.toarray()
        return Sigma1

    def _reduce_for_oneshot(self, estimators):
        worker_df = np.array(estimators).reshape(-1, 4).T
        num_in_worker_list = worker_df[0].tolist()
        theta_list = worker_df[1].tolist()
        sigma_inv_list = worker_df[2].tolist()
        t = np.mean(worker_df[3].tolist())
        worker_num = len(theta_list)
        num_of_variable = len(theta_list[0])
        reduce_start = time.time()
        for i in range(worker_num):
            if not i:
                theta_oneshot = theta_list[i]
            else:
                theta_oneshot = theta_oneshot + theta_list[i]
        theta_oneshot = 1 / worker_num * theta_oneshot
        reduce_end = time.time()
        self.oneshot_t = t + reduce_end - reduce_start
        self.oneshot_rho = theta_oneshot[0]
        self.oneshot_beta = theta_oneshot[1:]




    def _main(self, RDD):
        if self.method == "onestep":
            # the first iteration on each worker
            first_estimators = RDD.mapPartitions(self._worker_estimate_algorithm).collect()
            # reduce on the master to get the one-step estimator
            self._reduce_for_oneshot(first_estimators)
            self._reduce_estimator(first_estimators)
            print("===== Start Inference =====")
            # the Sigma1_k on each worker
            first_sigma1s = RDD.mapPartitions(self._worker_calculate_sigma1).collect()
            print("===============================")
            # reduce on the master to get the one-step SE
            s = time.time()
            onestep_Sigma1 = self._reduce_se(first_sigma1s, self.onestep_beta)
            onestep_Sigma2_inv = np.linalg.inv(self.onestep_Sigma2)
            self.onestep_se = (np.diag(onestep_Sigma2_inv @ onestep_Sigma1 @ onestep_Sigma2_inv)) ** 0.5
            e = time.time()
            print(e-s)
            print("===============================")
        elif self.method == "twostep":
            self.method = "onestep"
            # the first iteration on each worker
            onestep_t1 = time.time()
            first_estimators = RDD.mapPartitions(self._worker_estimate_algorithm).collect()
            # reduce on the master to get the one-step estimator
            # self._reduce_for_oneshot(first_estimators)
            onestep_t2 = time.time()
            self._reduce_estimator(first_estimators)
            onestep_t3 = time.time()
            print("===============================")
            # the Sigma1_k on each worker
            first_sigma1s = RDD.mapPartitions(self._worker_calculate_sigma1).collect()
            onestep_t4 = time.time()
            # reduce on the master to get the one-step SE
            self.onestep_Sigma1 = self._reduce_se(first_sigma1s, self.onestep_beta)
            onestep_Sigma2_inv = np.linalg.inv(self.onestep_Sigma2)
            self.onestep_se = np.diag(onestep_Sigma2_inv @ self.onestep_Sigma1 @ onestep_Sigma2_inv) ** 0.5
            onestep_t5 = time.time()
            print("onestep 时间")
            print(onestep_t1, onestep_t2, onestep_t3, onestep_t4, onestep_t5)
            print("===============================")
            # Reset some parameters
            self.method = "twostep"
            # the second iteration on each worker
            twostep_t1 = time.time()
            second_estimators = RDD.mapPartitions(self._worker_estimate_algorithm).collect()
            # reduce on the master to get the two-step estimator
            twostep_t2 = time.time()
            self._reduce_estimator(second_estimators)
            print("===============================")
            # the Sigma1_k on each worker
            twostep_t3 = time.time()
            second_sigma1s = RDD.mapPartitions(self._worker_calculate_sigma1).collect()
            twostep_t4 = time.time()
            # reduce on the master to get the one-step SE
            self.twostep_Sigma1 = self._reduce_se(second_sigma1s, self.twostep_beta)
            twostep_Sigma2_inv = np.linalg.inv(self.twostep_Sigma2)
            self.twostep_se = (np.diag(twostep_Sigma2_inv @ self.twostep_Sigma1 @ twostep_Sigma2_inv)) ** 0.5
            twostep_t5 = time.time()
            print("twostep 时间")
            print(twostep_t1, twostep_t2, twostep_t3, twostep_t4, twostep_t5)
            print("===============================")
        elif self.method == "onlyforsimulation":
            self.method = "onestep"
            # the first iteration on each worker
            t1 = time.time()
            first_estimators = RDD.mapPartitions(self._worker_estimate_algorithm).collect()
            t2 = time.time()
            self._reduce_for_oneshot(first_estimators)
            t3 = time.time()
            self._reduce_estimator(first_estimators)
            t4 = time.time()

            self.method = "twostep"
            second_estimators = RDD.mapPartitions(self._worker_estimate_algorithm).collect()
            t5 = time.time()
            self._reduce_estimator(second_estimators)
            t6 = time.time()
            print("时间")
            print(t1, t2, t3, t4, t5, t6)
            self.method = "onlyforsimulation"

if __name__ == '__main__':
    s1 = time.time()
    conf = SparkConf(). \
    setAll([('spark.pyspark.driver.python', './pyspark_env/pyspark_env/bin/python3'),
            ('spark.pyspark.python', './pyspark_env/pyspark_env/bin/python3')])
    sc = SparkContext.getOrCreate(conf)
    print(sys.executable)
    data_num = 945140
    worker_num = 100
    SigmaX = np.array(
        [[945140.,  88213.91825094, 177804.93426685, 100216.16547237, 446013.15801775],
       [ 88213.91825094, 945140.00000134, 833676.3726788 , 831388.94360036, 151028.36913732],
       [177804.93426685, 833676.3726788 , 945140.00000476, 819088.12732402, 286983.32534827],
       [100216.16547237, 831388.94360036, 819088.12732402, 945139.99999453, 180633.10278206],
       [446013.15801775, 151028.36913732, 286983.32534827, 180633.10278206, 945139.99999494]])
    # data = np.load("/data1/yelp/yelp_user_network_sar/test_data.npy", allow_pickle=True).tolist()
    # data = sp.load_npz("/data1/yelp/yelp_user_network_sar/All_estimate_data.npz").tocsc()
    print("------ 加载数据 ------")
    # data_list = np.load("/root/All_data_include_inference_100partition_v1.npy",
    #                     allow_pickle=True)
    data_list = np.load("/root/All_estimate_data_100partition_v1.npy", allow_pickle=True)
    omega_vec = sp.load_npz("/root/omega_vec.npz").tocsc()
    print("------ 加载数据完成 ------")
    for _ in range(10):
        real_case = SparseDistributedSAR(
        mode="real",
        extra_params={
            # "omega_vec": omega_vec.T.toarray()[0],
            "omega_vec": omega_vec,
            "data_num": data_num,
            "worker_num": worker_num,
            "rho_init": 0.,
            "beta_init": np.array([0., 0., 0., 0., 0.]),
            "seed1": 2021,
            "seed2": 2022,
            "proj_const": 20,
            "method": "onlyforsimulation",
            "sigma_X": SigmaX,
            "sigma_eps": 1,
            }
        )
        s2 = time.time()
        data_rdd = sc.parallelize(data_list, worker_num)
        real_case._main(data_rdd)
        s3 = time.time()
        print("启动Spark+读取数据:{};总时间:{}".format(s2-s1, s3-s1))
        print("-------------------------------------")
        print(real_case.onestep_rho)
        print(real_case.onestep_beta)
        # print(real_case.onestep_se)
        print(real_case.onestep_t)
        print("-------------------------------------")
        print(real_case.twostep_rho)
        print(real_case.twostep_beta)
        # print(real_case.twostep_se)
        print(real_case.twostep_t)
        print("-------------------------------------")
        print(real_case.oneshot_rho)
        print(real_case.oneshot_beta)
    # print(real_case.oneshot_t)

