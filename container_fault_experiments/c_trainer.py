"""
# @author qumu
# @date 2022/4/28
# @module trainer

trainers for attribution
"""
import numpy as np
import pandas as pd
import sklearn.linear_model as skl
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Lasso
# from badsql_tars.BMFS_pos_coef import BayesMulticolinearFeatureSelection
# from badsql_tars.BMFS_new_prior import BayesMulticolinearFeatureSelection
# from badsql_tars.BMFS_new_prior_optimized import BayesMulticolinearFeatureSelection
# from badsql_tars.BMFS_hopefully_final import BayesMulticolinearFeatureSelection
from BMFS import BayesMulticolinearFeatureSelection
#from badsql_tars import lava
#from badsql_tars.vblasso import VBLasso
from scipy import linalg
import os
import subprocess



def standardize(df):
    """
    标准化，zero mean, one std
    """
    EPS = 1e-16
    if isinstance(df, pd.DataFrame):
        df1 = (df - df.mean(axis=0)) / (df.std(axis=0) + EPS)
    elif isinstance(df, pd.Series):
        df1 = (df - df.mean(axis=0)) / (df.std() + EPS)
    return df1


def center_and_allstd(df):
    """中心化， zero mean"""
    EPS = 1e-16
    if isinstance(df, pd.DataFrame):
        df1 = df - df.mean(axis=0) / (np.std(df.values) + EPS)
    elif isinstance(df, pd.Series):
        df1 = df - df.mean() / (np.std(df.values) + EPS)
    return df1


def bayesFS(X, y, positive, normalize=None):
    model = skl.LinearRegression()
    # use all data to compute the prior
    X_std = standardize(X)
    fs_model = BayesMulticolinearFeatureSelection(X_prior=X.copy(), normalize=normalize)
    X_new = X.copy()
    y_new = y.copy()
    #X_new = pd.concat([X.copy()[:-1].reset_index(drop=True), X.copy()[1:].reset_index(drop=True)], axis=1)
    #y_new = y.copy()[1:]
    # use abnormal data (or data close to the abnormal point) to compute beta
    beta_est = fs_model.fit(X_new, y_new, max_iter=1000, tol_ll=1e-5, positive=positive)
    #model.coef_ = np.max(np.absolute(beta_est.numpy().reshape((-1, 2))), axis=1)
    model.coef_ = beta_est.numpy()
    return model


def fsMTS_train(X, y, normalize=None):
    X = X.interpolate()
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

    Xy = pd.concat([X_new[:-1].reset_index(drop=True), y_new[1:].reset_index(drop=True)], axis=1)
    # print(Xy)
    Xy.to_csv('Xy.csv')
    subprocess.call(['Rscript', '../fsMTS_train.R'])
    model = skl.LinearRegression()
    model.coef_ = pd.read_csv('coef.csv').values[:, 0]
    # print(model.coef_)
    return model


def my_ard_train(X, y, positive, normalize=None):
    model = skl.LinearRegression()
    # use all data to compute the prior
    #X_new = pd.concat([X.copy()[:-1].reset_index(drop=True), X.copy()[1:].reset_index(drop=True)], axis=1)
    #y_new = y.copy()[1:]
    fs_model = BayesMulticolinearFeatureSelection(K_prior=np.eye(X.shape[1]), normalize=normalize)
    # use abnormal data (or data close to the abnormal point) to compute beta
    beta_est = fs_model.fit(X, y, max_iter=1000, tol_ll=1e-5, positive=positive)
    model.coef_ = beta_est.numpy()
    #model.coef_ = np.max(np.absolute(beta_est.numpy().reshape((-1, 2))), axis=1)
    return model


def _alpha_grid(
    X,
    y,
    l1_ratio=1.0,
    eps=1e-3,
    n_alphas=30,
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

def my_enet_train(X, y, pos, normalize=None):
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

    #X_new = pd.concat([X_new.copy()[:-1].reset_index(drop=True), X_new.copy()[1:].reset_index(drop=True)], axis=1)
    #y_new = y_new.copy()[1:]
    alphas = _alpha_grid(X_new, y_new, l1_ratio=0.5)
    model = GridSearchCV(ElasticNet(l1_ratio=0.5, positive=pos), param_grid={'alpha': alphas, 'l1_ratio': [0.1, 0.5, 0.9]}, cv=5, n_jobs=4).fit(X_new, y_new).best_estimator_
    #model.coef_ = np.max(np.absolute(model.coef_.reshape((-1, 2))), axis=1)
    return model


def my_lasso_train(X, y, pos, normalize=None):
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
    #X_new = pd.concat([X_new.copy()[:-1].reset_index(drop=True), X_new.copy()[1:].reset_index(drop=True)], axis=1)
    #y_new = y_new.copy()[1:]
    alphas = _alpha_grid(X_new, y_new)
    model = GridSearchCV(Lasso(positive=pos), param_grid={'alpha': alphas}, cv=5).fit(X_new, y_new).best_estimator_
    
    #model.coef_ = np.max(np.absolute(model.coef_.reshape((-1, 2))), axis=1)
    return model
