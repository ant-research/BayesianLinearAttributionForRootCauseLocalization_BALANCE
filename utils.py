"""
# @author qumu
# @date 2022/4/28
# @module utils

utils functions
"""
import numpy as np



def shift(arr, num, fill_value=np.nan):
    """
    shift an array
    @param arr: list or 1darray
    @param num: shift length
    @param fill_value: default nan
    @return: new 1darray
    """
    # new array with dtype float
    result = arr.astype('float')

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def is_anomaly(ts, bound=1.5):
    y = ts
    q50 = np.percentile(y, 50)
    q75 = np.percentile(y, 75)
    q25 = np.percentile(y, 25)
    IQR = q75 - q25
    upper = q50 + bound * IQR
    print("median: {}, IQR: {}, upper: {}".format(q50, IQR, upper))
    return (y > upper).any()


def is_all_zeros_or_nans(series):
    return (series == 0.).sum() + series.isna().sum() == len(series)


def anomaly_detect(ts, bound=1.5):
    y = ts
    q50 = np.nanpercentile(y, 50)
    q75 = np.nanpercentile(y, 75)
    q25 = np.nanpercentile(y, 25)
    IQR = q75 - q25
    upper = q50 + bound * IQR
    #print("median: {}, IQR: {}, upper: {}".format(q50, IQR, upper))
    return upper


