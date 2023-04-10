#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 20:42:24 2022

@author: hugo
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
import os
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from BMFS import BayesMulticolinearFeatureSelection
from trainer import _alpha_grid
from sklearn.model_selection import GridSearchCV
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

def BMFS(X, y, K_prior=None):
    # use all data to compute the prior
    fs_model = BayesMulticolinearFeatureSelection(K_prior=K_prior, X_prior=X.copy(), normalize=None) # "allstd"
    X_new = X.copy()
    y_new = y.copy()
    beta_est = fs_model.fit(X_new, y_new, max_iter=10000, tol=1e-3, positive=False)
    return beta_est.numpy()


def mycorr(mat):
    return pd.DataFrame(mat).corr()

def sparse_rand_vector(p, pt):
    """

    @param p: dimension number
    @param pt: number of nonzeros
    @return: a (p,) shape array with pt nonzero random  entries
    """
    # coefs = np.random.binomial(1, 0.05, p).reshape(p)
    # if np.sum(coefs) == 0:
    #     coefs[np.random.choice(p, 1)[0]] = 1
    coefs = np.zeros(p)
    coefs[np.random.permutation(p)[:pt]] = 1
    return coefs

def sparse_rand_square_matrix(p):
    m = np.random.binomial(1, 0.05, p*p).reshape(p, p)
    if np.sum(m) == 0:
        m[np.random.choice(p, 1)[0], np.random.choice(p, 1)[0]] = 1
    return m


def gen_sparse_winv(p, pt):
    """

    Parameters
    ----------
    p: dimension or number of variables
    n: sample size
    pt: number of non-zero elements in the Cholesky decomposition of the precision matrix

    Returns
    -------
    Data and true precision matrix

    """

    idl_r, idl_c = np.tril_indices(p)
    pe = idl_r.size
    ide = np.random.permutation(pe)[:pt]

    Ku = np.zeros((p, p))
    Ku[idl_r[ide], idl_c[ide]] = np.sign(np.random.rand(pt) - 0.5) * (0.5 + 0.5 * np.random.rand(pt))
    np.fill_diagonal(Ku, 1)

    Su = np.linalg.inv(Ku)
    Sd = np.sqrt(np.sum(Su ** 2, axis = 1))
    Su = (1 / Sd)[:, None] * Su
    Ku = Ku * Sd
    Ktrue = Ku.T @ Ku

    id = np.random.permutation(p)
    Ktrue = Ktrue[np.ix_(id, id)]

    return Ktrue


def generate_arti_data(n, p, p_small, percent_nonzero, noise_std, perfectly_correlated = False):
    """
    y = X * beta + epsilon
    
    X = Z * Q
    beta = inv(Q) * alpha
    
    Z:      n x p_small matrix
    alpha:  p_small x 1 vector
    
    """
    
    # random partiion of p into p_small block
    alpha = np.zeros(p_small)
    
    par_length = np.zeros(p_small, dtype = int)
    total_length = 0
    nonzero_length = 0
    for i in range(p_small - 1):
        par_length[i] = np.random.randint(1, p - p_small + i - total_length + 2)
        total_length += par_length[i]
        nonzero_length += par_length[i]
        if i == 0:
            id_col = np.zeros(par_length[i], dtype = int)
        else:
            id_col = np.r_[id_col, i * np.ones(par_length[i], dtype = int)]
        if nonzero_length / p <= percent_nonzero:
            alpha[i] = (np.sign(np.random.randn(1)) * (0.1 + np.random.rand(1)))
        else:
            nonzero_length -= par_length[i]
    par_length[-1] = p - total_length
    id_col = np.r_[id_col, (i + 1) * np.ones(par_length[-1], dtype = int)]
    nonzero_length += par_length[-1]
    if nonzero_length / p <= percent_nonzero:
        alpha[-1] = (np.sign(np.random.randn(1)) * (0.1 + np.random.rand(1)))
    
    
    
    # generate alpha and Z
    Z = np.random.randn(n, p_small)  # ~N(0, 1)
    # alpha = (np.sign(np.random.randn(p_small)) * (0.1 + np.random.rand(p_small)))
    # alpha[np.random.permutation(p_small)[:int(p_small * percent_zero)]] = 0
    y = Z @ alpha + noise_std * np.random.randn(n)
    
    
    
    # generate p x p_small orthonormal matrix
    W = np.zeros((p, p_small))
    if perfectly_correlated:
        W[np.arange(p), id_col] = np.sign(np.random.randn(p))
    else:
        W[np.arange(p), id_col] = np.random.randn(p)
    Q, _ = np.linalg.qr(W)
    Q /= np.sqrt(np.sum(Q ** 2, axis = 1))[:, None]
    
    # generate X and beta
    X = Z @ Q.T
    beta = Q @ alpha
    
    return pd.DataFrame(X), pd.Series(y), beta

def generate_perfect_collinear_dataset(n, p, pt):
    p_group = np.random.randint(10, 100)
    Z = np.random.randn(n, p_group)  # ~N(0, 1)
    beta_group = (np.sign(np.random.randn(p_group)) * (0.1 + np.random.rand(p_group))) * sparse_rand_vector(p_group, pt)
    id_repeat = np.sort(np.random.randint(p_group, size = p))
    X = Z[:, id_repeat]
    beta = beta_group[id_repeat]
    y = np.dot(X, beta) + 1e-2 * np.random.randn(n)
    return pd.DataFrame(X), pd.Series(y), beta


from sklearn.linear_model import LassoCV, ElasticNetCV, ARDRegression, Lasso, ElasticNet
def micro_f_score(beta, beta_est):
    true_set= set(np.flatnonzero(np.abs(beta) > 0))
    pos_set = set(np.flatnonzero(np.abs(beta_est) > 0))
    neg_set = set(np.flatnonzero(np.abs(beta_est) <= 0))
    TP = len(true_set.intersection(pos_set))
    FP = len(pos_set.difference(true_set))
    TN = len(neg_set.difference(pos_set))
    FN = len(true_set.difference(pos_set))
    precision = TP / (TP + FP + 1e-4)
    recall = TP / (TP + FN + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)
    # print(true_set)
    # print(pos_set)
    return precision, recall, f1

@ignore_warnings(category=ConvergenceWarning)
def evaluate_all_once(n, p, p_small, percent_nonzero, noise_std, perfectly_correlated, percent_missing=0.):
    print("n={}, p={}, p_small={},  percent_nonzero={}, noise_std={}, perfectly={}, percent_missing={}".format(n, p, p_small,  percent_nonzero, noise_std, perfectly_correlated, percent_missing))
    result = {}
    ard_res, lasso_res, enet_res, bmfs_res = [], [], [], []
    for i in range(100):
        X_all, y, beta = generate_arti_data(10*n, p, p_small, percent_nonzero, noise_std, perfectly_correlated)  #
        X = X_all[:n].copy()
        y = y[:n]
        # X = (X - X.mean(axis=0)) / X.std(axis=0)
        # y = (y - y.mean()) / y.std()
        missing_num = int(n * percent_missing)
        # print(missing_num)
        # print(np.random.choice(n, missing_num, replace=False))
        for k in np.arange(p):
            X.values[np.random.choice(n, missing_num, replace=False), k] = np.nan



        start = time.time()
        # ard_beta = ARDRegression(fit_intercept = False).fit(X, y).coef_   #, threshold_lambda = 1e5
        fs_model = BayesMulticolinearFeatureSelection(K_prior=np.eye(p), normalize=None)  # "allstd" #
        X_new = X.copy()
        y_new = y.copy()
        ard_beta = fs_model.fit(X_new, y_new, max_iter=1000, tol_ll=1e-3,  positive=False).numpy()
        tmp = {}
        tmp['id'] = i
        tmp['running_time'] = time.time() - start
        tmp['mse'] = np.mean((ard_beta - beta) ** 2)  # np.mean((np.dot(X.values, ard_beta) - y.values)**2)
        tmp['precision'], tmp['recall'], tmp['f-score'] = micro_f_score(beta, ard_beta)
        ard_res.append(tmp)
        print("ARD result:", tmp)
        
        start = time.time()
        # lasso_beta = LassoCV(cv=5, fit_intercept = False).fit(X, y).coef_
        X_new = X.copy()
        X_new.fillna(0., inplace=True)
        lasso_beta = GridSearchCV(Lasso(fit_intercept=False), param_grid={'alpha': np.logspace(-2, 2, 30)},
                                  cv=5).fit(X_new, y).best_estimator_.coef_

        tmp = {}
        tmp['id'] = i
        tmp['running_time'] = time.time() - start
        tmp['mse'] = np.mean((lasso_beta - beta) ** 2)  # np.mean((np.dot(X.values, lasso_beta) - y.values)**2)
        tmp['precision'], tmp['recall'], tmp['f-score'] = micro_f_score(beta, lasso_beta)
        lasso_res.append(tmp)
        print("lasso result:", tmp)

        start = time.time()
        X_new = X.copy()
        X_new.fillna(0., inplace=True)
        # enet_beta = ElasticNetCV(cv=5, fit_intercept = False).fit(X, y).coef_
        enet_beta = GridSearchCV(ElasticNet(fit_intercept=False), param_grid={'alpha': np.logspace(-2, 2, 10), 'l1_ratio': [0.1, 0.5, 0.9]}, cv=5).fit(X_new, y).best_estimator_.coef_
        tmp = {}
        tmp['id'] = i
        tmp['running_time'] = time.time() - start
        tmp['mse'] = np.mean((enet_beta - beta) ** 2)  ##np.mean((np.dot(X.values, enet_beta) - y.values)**2)
        tmp['precision'], tmp['recall'], tmp['f-score'] = micro_f_score(beta, enet_beta)
        enet_res.append(tmp)
        print("enet result:", tmp)

        start = time.time()
        # bmfs_beta = BMFS(X, y, K_prior = np.eye(p)) #  #  W.T @ W / np.std(X.values)
        fs_model = BayesMulticolinearFeatureSelection(X_prior=X.copy(),
                                                      normalize='standard')  # "allstd" # K_prior = np.eye(p),
        X_new = X.copy()
        y_new = y.copy()
        bmfs_beta = fs_model.fit(X_new, y_new, max_iter=1000, tol_ll=1e-3, positive=False).numpy()
        tmp = {}
        tmp['id'] = i
        tmp['running_time'] = round(time.time() - start, 2)
        tmp['mse'] = np.mean((bmfs_beta - beta) ** 2)  ##np.mean((np.dot(X.values, bmfs_beta) - y.values)**2)
        tmp['precision'], tmp['recall'], tmp['f-score'] = micro_f_score(beta, bmfs_beta)
        # print(tmp['f-score'])
        print("BMFS result:", tmp)
        bmfs_res.append(tmp)
        print(bmfs_beta)

    result['ARD'] = pd.DataFrame(ard_res).set_index('id')
    result['Lasso'] = pd.DataFrame(lasso_res).set_index('id')
    result['ElasticNet'] = pd.DataFrame(enet_res).set_index('id')
    result['BMFS'] = pd.DataFrame(bmfs_res).set_index('id')

    res_df = pd.concat(result, axis=1)
    print(res_df.mean())
    return res_df.mean()


def experiment_1():
    """
    fix n, change p, compare different multicollinearity
    @return:
    """
    n = 100
    plist = [20, 50, 100, 200, 500, 1000]
    ex1 = {}
    ex2 = {}
    ex3 = {}

    res1 = []
    # example1,fix n, change p & indpendent p_small = p, perfectly_correlated=False
    for p in plist:
        res = evaluate_all_once(n, p, p, percent_nonzero=5/p, noise_std=0.1, perfectly_correlated=False)
        ex1["p={}".format(p)] = res
    res1 = pd.concat(ex1, axis=1)
    print(res1)

    # example2,fix n, change p & indpendent p_small = 0.1 * p, perfectly_correlated=False
    for p in plist:
        res = evaluate_all_once(n, p, max(5, int(0.5 * p)), percent_nonzero=5/p, noise_std=0.1, perfectly_correlated=False)
        ex2["p={}".format(p)] = res
    res2 = pd.concat(ex2, axis=1)
    print(res2)

    # example3,fix n, change p & indpendent p_small = 0.1 * p, perfectly_correlated=True
    for p in plist:
        res = evaluate_all_once(n, p, max(5, int(0.4 * p)), percent_nonzero=5 / p, noise_std=0.1, perfectly_correlated=True)
        ex3["p={}".format(p)] = res

    res3 = pd.concat(ex3, axis=1)
    print(res3)
    return res1, res2, res3


def experiment_2():
    """
    fix n and p, fix partial collinear, close to real data
    change noise_std
    @return:
    """
    n = 100
    p = 1000
    noise_list = [0.01, 0.1, 1, 3]
    ex = {}
    # example1,fix n, change p & indpendent p_small = p, perfectly_correlated=False
    for noise_std in noise_list:
        res = evaluate_all_once(n, p, 400, percent_nonzero=5/p, noise_std=noise_std, perfectly_correlated=False)
        ex["noise_std={}".format(noise_std)] = res


    res = pd.concat(ex, axis=1)

    return res


def experiment_3():
    """
    fix n and p, fix partial collinear, close to real data
    change noise_std
    @return:
    """
    n = 100
    p = 1000
    percent_nonzeros = [2/p, 5/p, 10/p, 20/p, 50/p, 100/p]
    # percent_nonzeros = [50 / p]
    ex = {}

    for percent_nonzero in percent_nonzeros:
        res = evaluate_all_once(n, p, 400, percent_nonzero=percent_nonzero, noise_std=0.1, perfectly_correlated=False)
        ex["percent_nonzero={}".format(percent_nonzero)] = res


    res = pd.concat(ex, axis=1)

    return res


def experiment_4():
    """
    fix n and p, fix partial collinear, close to real data
    change noise_std
    @return:
    """
    n = 100
    p = 1000
    percent_missings = [0.1, 0.2, 0.3, 0.4, 0.5]

    ex = {}

    for percent_missing in percent_missings:
        res = evaluate_all_once(n, p, 400, percent_nonzero=5/p, noise_std=0.1, perfectly_correlated=False, percent_missing=percent_missing)
        ex["percent_missing={}".format(percent_missing)] = res


    res = pd.concat(ex, axis=1)

    return res

if __name__ == '__main__':
    print("TEST 1")
    res1, res2, res3 = experiment_1()
    res1.to_csv("t1_o_1.csv")
    res2.to_csv("t1_o_2.csv")
    res3.to_csv("t1_o_3.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res1)
        print(res2)
        print(res3)

    print("TEST 2")
    t2_res = experiment_2()
    t2_res.to_csv("t2o.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(t2_res)

    print("TEST 3")
    t3_res = experiment_3()
    t3_res.to_csv("t3o.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(t3_res)

    print("TEST 4")
    t4_res = experiment_4()
    t4_res.to_csv("t4o.csv")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(t4_res)
