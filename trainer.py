"""
# @author qumu
# @date 2022/4/28
# @module trainer

trainers for attribution
"""
import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso
from BMFS import BayesMulticolinearFeatureSelection
from scipy import linalg
import os
import subprocess
import warnings
# ----------------settings------------------
warnings.filterwarnings("ignore")


def standardize(df):
    """
    zero mean, one std
    """
    EPS = 1e-16
    if isinstance(df, pd.DataFrame):
        df1 = (df - df.mean(axis=0)) / (df.std(axis=0) + EPS)
    elif isinstance(df, pd.Series):
        df1 = (df - df.mean(axis=0)) / (df.std() + EPS)
    return df1


def center_and_allstd(df):
    """ zero mean, all std"""
    EPS = 1e-16
    if isinstance(df, pd.DataFrame):
        df1 = df - df.mean(axis=0) / (np.std(df.values) + EPS)
    elif isinstance(df, pd.Series):
        df1 = df - df.mean() / (np.std(df.values) + EPS)
    return df1


def bayesFS(X: pd.DataFrame, y: pd.Series, positive, tol=1e-2, tol_ll=1e-3,  normalize=None):
    """
    BMFS trainer
    @param X:
    @param y:
    @param positive:
    @param normalize:
    @return:
    """
    model = skl.LinearRegression()
    # use all data to compute the prior
    fs_model = BayesMulticolinearFeatureSelection(X_prior=X.copy(), normalize=normalize)
    X_new = X.copy()
    y_new = y.copy()
    # use abnormal data (or data close to the abnormal point) to compute beta
    beta_est = fs_model.fit(X_new, y_new, max_iter=1000, tol=tol, tol_ll=tol_ll, positive=positive)
    model.coef_ = beta_est.numpy()
    return model


def enet_train(X, y, normalize=None, **kwargs):
    """E-Net Trainer with sklearn ElasticNetCV"""
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y

    model = skl.ElasticNetCV(l1_ratio=[0.1], cv=5, n_jobs=4, **kwargs).fit(X_new, y_new)
    # mask = model.coef_ > 0
    # # print(X.mean())
    # if np.sum(mask) > 0:
    #     X_filter = X_new.loc[:, mask]
    #     model.coef_[~mask] = 0
    #     model.coef_[mask] = skl.LinearRegression(positive=True).fit(X_filter, y_new).coef_
    return model


def lasso_train(X, y, normalize=None, **kwargs):
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y
    model = skl.LassoCV(cv=5, n_jobs=4, **kwargs).fit(X_new, y_new)
    return model


def ard_regression_train(X, y, normalize=None):
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y
    model = skl.ARDRegression().fit(X_new, y_new)
    return model


def fsMTS_train(X, y, normalize=None):
    """fsMTS Trainer with fsMTS.R calling public R package fsMTS """
    X = X + 1e-8 * np.random.randn(*X.shape)
    y = y + 1e-8 * np.random.randn(*y.shape)
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y

    Xy = pd.concat([X_new[:-2].reset_index(drop=True), y_new[2:].reset_index(drop=True)], axis=1)
    # print(Xy)
    Xy.to_csv('Xy.csv')
    subprocess.call(['Rscript', '../fsMTS_train.R'])
    model = skl.LinearRegression()
    model.coef_ = pd.read_csv('coef.csv').values[:, 0]
    # print(model.coef_)
    return model


def my_ard_train(X, y, positive, normalize=None):
    """ARD Trainer, same base implementation as BMFS but different priors"""
    model = skl.LinearRegression()
    # use all data to compute the prior
    fs_model = BayesMulticolinearFeatureSelection(K_prior=np.eye(X.shape[1]), normalize=normalize)
    X_new = X.copy()
    y_new = y.copy()
    # use abnormal data (or data close to the abnormal point) to compute beta
    beta_est = fs_model.fit(X_new, y_new, max_iter=1000, tol_ll=1e-3, positive=positive)
    model.coef_ = beta_est.numpy()
    return model


def _alpha_grid(
    X,
    y,
    l1_ratio=1.0,
    eps=1e-3,
    n_alphas=100,
):
    """Compute the grid of alpha values for elastic net parameter search
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication
    y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        Target values
    Xy : array-like of shape (n_features,) or (n_features, n_outputs),\
         default=None
        Xy = np.dot(X.T, y) that can be precomputed.
    l1_ratio : float, default=1.0
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.
    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``
    n_alphas : int, default=100
        Number of alphas along the regularization path
    """
    if l1_ratio == 0:
        raise ValueError(
            "Automatic alpha grid generation is not supported for"
            " l1_ratio=0. Please supply a grid by providing "
            "your estimator with the appropriate `alphas=` "
            "argument."
        )
    n_samples = len(y)

    Xy = np.dot(X.values.T, y.values)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    alpha_max = np.sqrt(np.sum(Xy**2, axis=1)).max() / (n_samples * l1_ratio)

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max), num=n_alphas)[
        ::-1
    ]


@ignore_warnings(category=ConvergenceWarning)
def my_enet_train(X, y, pos, normalize=None):
    """E-Net Trainer with sklearn ElasticNet and grid search, search grid from rules same as sklearn implementation"""
    X = X.interpolate()
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y

    alphas = _alpha_grid(X, y, l1_ratio=0.5, n_alphas=30)
    # model = GridSearchCV(ElasticNet(l1_ratio=0.5, positive=pos), param_grid={'alpha': alphas}, cv=5, n_jobs=4).fit(X_new, y_new).best_estimator_
    model = GridSearchCV(ElasticNet(), param_grid={'alpha': alphas, 'l1_ratio': [0.1, 0.5, 0.9]}, cv=5, n_jobs=4).fit(X_new, y_new).best_estimator_
    return model


@ignore_warnings(category=ConvergenceWarning)
def my_lasso_train(X, y, pos, normalize=None):
    """Lasso Trainer with sklearn Lasso and grid search, search grid from rules same as sklearn implementation"""
    X = X.interpolate()
    if normalize == "standard":
        X_new = standardize(X)
        y_new = standardize(y)
    elif normalize == "allstd":
        X_new = center_and_allstd(X)
        y_new = center_and_allstd(y)
    else:
        X_new = X
        y_new = y
    alphas = _alpha_grid(X, y, n_alphas=30)
    model = GridSearchCV(Lasso(positive=pos), param_grid={'alpha': alphas}, cv=5).fit(X_new, y_new).best_estimator_
    return model
