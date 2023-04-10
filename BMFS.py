#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian Multicollinear Feature Selection
"""

import torch

torch.set_default_dtype(torch.float64)
from scipy import special as sp
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
# from KDEpy import FFTKDE
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture as BayesGMM
# from mpmath import gammainc
from GMM import GMM
# from sympy import Float
import sys


class BayesMulticolinearFeatureSelection:

    def __init__(self, K_prior=None, X_prior=None, normalize='standard'):  #
        """
        Gain prior knowledge about the correlation between features
        K_prior: (p, p) tensor of prior covariance of the correlated features
        X_prior: (n, p) tensor of feature vectors, where n is the sample size and p is the dimension
        If K_prior is not None, user specified K_prior is used; otherwise, we compute K_prior from X_prior.
        normalize: specify the normalization method. If normalize = 'standard', we normalize each feature individually; 
        if normalize = 'allstd', we normalize all features simultaneously.
        """
        self.normalize = normalize
        if K_prior is None:
            X_prior -= X_prior.mean()
            if self.normalize == 'allstd':
                # print(self.normalize)
                X_prior /= np.nanstd(X_prior.values)
            elif self.normalize == 'standard':
                # print(self.normalize)
                X_prior /= X_prior.std()
            X_std = X_prior.std().values
            X_std[np.isnan(X_std)] = 1
            X_prior.fillna(0, inplace=True)  # !!!!
            X_corr = X_prior.corr().values
            X_corr[np.isnan(X_corr)] = 0
            np.fill_diagonal(X_corr, 1)
            X_cov = X_std[:, None] * (0.9 * X_corr + 0.1 * np.eye(X_prior.shape[1])) * X_std[None, :]
            self.K_prior = torch.tensor(X_cov)  # * X_prior.shape[0]
        else:
            self.K_prior = torch.tensor(K_prior)

    def compute_upincgammad_expd(self, c, d):
        """
        compute Gamma(c, d) * exp(d), where Gamma denotes the upper incomplete gamma function
        cf. Footnote 1 in Page 7
        """
        return sp.hyperu(1 - c, 1 - c, d)

    def p_mu_lbd_p_d(self):
        """
        compute the derivative of <lambda> (see Eq-(23) for definition) w.r.t. d
        """
        return (self.lbd - 1 / self.d) / (self.d * self.exp1d_m_expd)

    def fit(self, X, y, alpha=None, eta0=0.9, max_iter=1000, tol=1e-2, tol_ll=1e-3, positive=False, verbose=True):
        """
        variational inference to solve BMFS
        X: (n, p) tensor of feature vectors, where n is the sample size and p is the dimension
        y: (n, ) tensor of predictors
        alpha: initialization of the inverse variance of the observation noise
        eta0: initial step size for Amijo's rule
        max_ter: maximum number of iterations
        tol: tolerance between every two consecutive updates of <beta> for convergence check
        tol_ll: toleraance between every two consecutive udpates of the shrinkage weight density for convergence check
        positive: if positive is True, beta can only take positive values
        verbose: if verbose is True, print messages for convergence check
        """
        if verbose:
            print("initialize BMFS")
        torch.manual_seed(0)

        # normalize data and convert them to tensors
        n, p = X.shape
        if self.normalize == 'allstd':
            X /= np.nanstd(X.values)
            y /= y.std()
        elif self.normalize == 'standard':
            X /= X.std()
            y /= y.std()

        X -= X.mean()
        y -= y.mean()

        X = torch.tensor(X.values)
        y = torch.tensor(y.values)

        # check whether there exist missing data
        id_nan_X = torch.isnan(X)
        id_nan_y = torch.isnan(y)

        X[id_nan_X] = 0
        y[id_nan_y] = 0
        id_nan_Xf = id_nan_X.to(torch.float64)
        id_nan_yf = id_nan_y.to(torch.float64)
        id_X_row = torch.nonzero(id_nan_Xf.sum(axis=1) > 0)[:, 0]

        # pre-compute constant tensors to be used below and initialize all variational parameters
        K_cond = X.T @ X
        h_cond = X.T @ y
        y2 = torch.sum(y ** 2)

        mu_beta_init = torch.linalg.solve(K_cond / n + 0.1 * torch.eye(p), h_cond / n)
        if positive:
            regr = ElasticNet(random_state=0, positive=True, alpha=0.1)
            mu_beta_init = torch.tensor(regr.fit(X, y).coef_) + 1e-4
            mu_log_beta = torch.log(mu_beta_init)
            var_log_beta = 1e-2 * torch.ones(mu_beta_init.shape[0])
            h_log_beta = mu_log_beta / var_log_beta
            zeta_log_beta = 1 / var_log_beta
            mu_beta = torch.exp(mu_log_beta + var_log_beta / 2)
            mu_beta2 = torch.exp(2 * (mu_log_beta + var_log_beta))

        if alpha is None:
            alpha = 1.0
        gamma = 10.0
        b = 100.
        self.d = gamma / 2 * mu_beta_init ** 2 + 1e-2  # torch.ones(p)
        self.exp1d_m_expd = self.compute_upincgammad_expd(0, self.d)
        self.lbd = self.compute_upincgammad_expd(-1, self.d) / self.exp1d_m_expd
        self.sqrt_lbd = sp.gamma(1.5) * self.compute_upincgammad_expd(-0.5, self.d) / self.exp1d_m_expd

        ll_old = self.lbd / (1 + self.lbd)

        # apply the update rules iteratively
        if verbose:
            print("BMFS begins ...")

        for i in range(max_iter):

            # update q_beta
            sqrt_lbdTlbd = self.sqrt_lbd.view(-1, 1) * self.sqrt_lbd.view(1, -1)
            sqrt_lbdTlbd.diagonal().copy_(self.lbd)
            J_beta = alpha * K_cond + gamma * self.K_prior * sqrt_lbdTlbd + 1e-2 * torch.eye(
                p)  # add a small identity matrix to guarantee the positive definiteness

            h_beta = alpha * h_cond

            if positive:
                J_beta_diag = J_beta.diag()
                J_beta.fill_diagonal_(0)

                tmp1 = J_beta_diag * mu_beta2
                tmp2 = J_beta @ mu_beta
                tmp3 = (h_beta - tmp2) * mu_beta

                elbo_init = - (torch.sum(tmp1) + mu_beta @ tmp2) / 2 + h_beta @ mu_beta + mu_log_beta.sum() - torch.log(
                    zeta_log_beta).sum() / 2
                grad_mu = tmp3 * (1 - mu_log_beta) - tmp1 * (1 - 2 * mu_log_beta)
                grad_mu2 = tmp3 / 2 - tmp1

                natgrad_h = grad_mu + 1 - h_log_beta
                natgrad_zeta = -2 * grad_mu2 - zeta_log_beta

                grad_h = grad_mu / zeta_log_beta + grad_mu2 * 2 * h_log_beta / zeta_log_beta ** 2 + 1 / zeta_log_beta
                grad_zeta = - grad_mu * h_log_beta / zeta_log_beta ** 2 - grad_mu2 * (
                            1 + 2 * h_log_beta ** 2 / zeta_log_beta) / zeta_log_beta ** 2 - \
                            1 / zeta_log_beta / 2 - mu_log_beta / zeta_log_beta

                prod_grad = (natgrad_h * grad_h + natgrad_zeta * grad_zeta).sum()
                # if prod_grad < 0:
                #     print("prod_grad is smaller than 0 when updating beta, please check the inputs")

                eta = eta0
                for j in range(100):
                    zeta_log_beta_update = zeta_log_beta + eta * grad_zeta
                    if zeta_log_beta_update.min() > 0:
                        h_log_beta_update = h_log_beta + eta * grad_h
                        mu_log_beta_update = h_log_beta_update / zeta_log_beta_update
                        var_log_beta_update = 1 / zeta_log_beta_update
                        mu_beta_update = torch.exp(mu_log_beta_update + var_log_beta_update / 2)
                        mu_beta2_update = torch.exp(2 * (mu_log_beta_update + var_log_beta_update))
                        elbo_update = - (torch.sum(
                            J_beta_diag * mu_beta2_update) + mu_beta_update @ J_beta @ mu_beta_update) / 2 + \
                                      h_beta @ mu_beta_update + mu_log_beta_update.sum() - torch.log(
                            zeta_log_beta_update).sum() / 2
                        if elbo_update >= elbo_init + 1e-4 * eta * prod_grad:
                            h_log_beta = h_log_beta_update.clone()
                            zeta_log_beta = zeta_log_beta_update.clone()
                            mu_log_beta = mu_log_beta_update.clone()
                            var_log_beta = var_log_beta_update.clone()
                            mu_beta = mu_beta_update.clone()
                            mu_beta2 = mu_beta2_update.clone()
                            break
                        else:
                            eta /= 2
                    else:
                        eta /= 2

                mu_betaTbeta = mu_beta.view(-1, 1) @ mu_beta.view(1, -1)
                mu_betaTbeta.diagonal().copy_(mu_beta2)

            else:
                try:
                    mu_beta = torch.linalg.solve(J_beta, h_beta)
                except:
                    print(J_beta.diag())
                cov_beta = torch.linalg.inv(J_beta)
                mu_betaTbeta = mu_beta.view(-1, 1) @ mu_beta.view(1, -1) + cov_beta

            # update q_lbd
            J_lbd = gamma * self.K_prior * mu_betaTbeta
            J_lbd_diag = J_lbd.diag()
            J_lbd.fill_diagonal_(0)

            r = sp.gamma(1.5) * (self.compute_upincgammad_expd(-0.5, self.d) - self.exp1d_m_expd / self.d.sqrt()) \
                / (self.compute_upincgammad_expd(-1, self.d) - self.exp1d_m_expd / self.d)

            tmp1 = J_lbd @ self.sqrt_lbd
            natgrad_d = tmp1 * r + J_lbd_diag / 2 - self.d

            prod_grad = - (natgrad_d ** 2 * self.p_mu_lbd_p_d()).sum()
            # if prod_grad < 0:
            #     prod_grad = - prod_grad
            #     print("prod_grad is negative")

            b_gamma = torch.sum(J_lbd_diag * self.lbd) + self.sqrt_lbd @ J_lbd @ self.sqrt_lbd
            elbo_init = - b_gamma / 2 + torch.sum(torch.log(sp.exp1(self.d))) + torch.sum(
                1 / self.exp1d_m_expd)  # torch.sum(self.d * (self.lbd + 1))

            # Amijo's rule for step size selection
            eta = eta0
            for j in range(100):
                d_update = self.d + eta * natgrad_d
                if d_update.min() > 0:  # d_update.min() > 1e-4 and d_update.max() < 1e4:

                    exp1d_m_expd = self.compute_upincgammad_expd(0, d_update)
                    lbd_update = self.compute_upincgammad_expd(-1, d_update) / exp1d_m_expd
                    sqrt_lbd_update = sp.gamma(1.5) * self.compute_upincgammad_expd(-0.5, d_update) / exp1d_m_expd

                    b_gamma_update = torch.sum(J_lbd_diag * lbd_update) + sqrt_lbd_update @ J_lbd @ sqrt_lbd_update
                    elbo_update = - b_gamma_update / 2 + torch.sum(torch.log(sp.exp1(d_update))) + torch.sum(
                        1 / exp1d_m_expd)

                    if elbo_update >= elbo_init + 1e-4 * eta * prod_grad:
                        self.d = d_update.clone()
                        self.exp1d_m_expd = exp1d_m_expd.clone()
                        self.lbd = lbd_update.clone()
                        self.sqrt_lbd = sqrt_lbd_update.clone()
                        b_gamma = b_gamma_update.clone()
                        break
                    else:
                        eta /= 2
                else:
                    eta /= 2

            # if torch.isnan(self.lbd).sum() > 0:
            #     self.lbd

            # if b_gamma < 0:
            #     print("b_gamma is negative")

            # update q_alpha
            alpha = n / (y2 + torch.sum(mu_betaTbeta * K_cond) - 2 * (h_cond @ mu_beta) + 1e-8 * n)
            # if alpha < 0:
            #     print("alpha is negative")

            # update q_gamma
            gamma *= p / (b_gamma + 1e-8 * p * gamma)
            # if gamma < 0:
            #     print("gamma is negative")

            # impute missing data
            J_X = mu_betaTbeta  # + torch.eye(mu_beta2.shape[0])
            for j in id_X_row:
                id_u = torch.nonzero(id_nan_X[j])[:, 0]
                id_o = torch.nonzero(~ id_nan_X[j])[:, 0]
                h_X = y[j] * mu_beta
                X[j, id_u] = torch.linalg.solve(J_X[np.ix_(id_u, id_u)],
                                                h_X[id_u] - J_X[np.ix_(id_u, id_o)] @ X[j, id_o])

            K_cond = X.T @ X
            h_cond = X.T @ y

            ll = self.lbd / (1 + self.lbd)
            diff_beta = (mu_beta - mu_beta_init).abs().mean()
            diff_ll = (ll - ll_old).abs().mean()
            if verbose:
                print(f'iter = {i}, mu_beta difference: {diff_beta}, ll difference: {diff_ll}')
            if diff_beta <= tol and diff_ll <= tol_ll:
                if verbose:
                    print("BMFS converges")
                break
            else:
                mu_beta_init = mu_beta.clone()
                ll_old = ll.clone()

        if verbose and (diff_beta > tol or diff_ll > tol_ll):
            print("BMFS reaches the maximum number of iterations")

        # soft thresholding and refitting
        if verbose:
            print("Begin soft thresholding")
        ll = ll.numpy()
        weights_init = np.r_[np.sum(ll < 0.4), np.sum(ll > 0.6)].astype(np.float64) + 1.
        weights_init /= np.sum(weights_init)

        gm = GMM(2, 1, init_mu=np.r_[np.min(ll), np.max(ll)][:, None], init_sigma=0.01 * np.ones((2, 1, 1)),
                 init_pi=weights_init)
        gm.init_em(ll[:, None])
        log_llh_old = np.inf
        for i in range(max_iter):
            gm.e_step()
            gm.m_step()
            log_llh = gm.log_likelihood(X)
            if np.abs(log_llh - log_llh_old) < tol:
                break
            else:
                log_llh_old = log_llh.copy()
        mu0, mu1 = np.squeeze(gm.mu[0]), np.squeeze(gm.mu[1])
        std0, std1 = np.sqrt(np.squeeze(gm.sigma[0])), np.sqrt(np.squeeze(gm.sigma[1]))
        x_axis = np.linspace(mu0, mu1, 100)
        y_axis = gm.pi[0] * norm(mu0, std0).pdf(x_axis) + gm.pi[1] * norm(mu1, std1).pdf(x_axis)

        thr = x_axis[np.argmin(y_axis)]
        id_nonzero = (ll <= thr)
        X_small = X[:, id_nonzero]
        mu_beta = torch.zeros(p)
        if np.sum(id_nonzero) <= n:
            mu_beta[id_nonzero] = torch.linalg.solve(X_small.T @ X_small + 1e-2 * torch.eye(X_small.shape[1]),
                                                     X_small.T @ y)
        else:
            mu_beta[id_nonzero] = torch.linalg.solve(X_small.T @ X_small + 0.1 * torch.eye(X_small.shape[1]),
                                                     X_small.T @ y)

        return mu_beta
