"""
# @author qumu
# @date 2022/4/28
# @module attribution

attribution method for linear model
"""
import pandas as pd


def gradients_(model, X, thd, reverse=True):
    """
    use gradients directly as attribution
    @param model: a model instance with model.coef_ as gradients
    @param X: regressor DataFrame
    @param thd:
    @param reverse:
    @return:
    """
    cols = X.loc[:, model.coef_ > thd].columns.tolist()
    coefs = model.coef_[model.coef_ > thd].tolist()
    sorted_gradients_res = sorted(zip(cols, coefs), key=lambda k: k[1], reverse=reverse)
    return [col for col, graident in sorted_gradients_res]


def gradients_x_inputs_(model, X, before, thd=0):
    """
    use gradient * input as attribution, a.k.a baseline=0
    @param model:
    @param X:
    @param before: target index in X
    @param thd:
    @return:
    """
    if before > len(X) - 1:
        before = len(X) - 1
    sorted_attributions = (X.loc[before, model.coef_ > thd] * model.coef_[model.coef_ > thd]).sort_values(ascending=False)
    sorted_attributions_res = sorted_attributions.index.tolist()
    return sorted_attributions_res


def deeplift_linear_(model, X: pd.DataFrame, before, thd=0):
    """
    use deeplift method (average gradient * (change of x from baseline))
    @param model:
    @param X:
    @param before:
    @param thd:
    @return:
    """
    if before > len(X) - 1:
        before = len(X) - 1
    baseline = before - 10
    sorted_attributions = ((X.loc[before, model.coef_ > thd] - X.loc[baseline, model.coef_ > thd]) * model.coef_[model.coef_ > thd]).sort_values(ascending=False)
    sorted_attributions_res = sorted_attributions.index.tolist()
    return sorted_attributions_res
