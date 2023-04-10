# -*- coding: utf-8 -*-
# @author qumu
# @date 2022/4/21
"""
# @author qumu
# @date 2022/4/28
# @module evaluate
"""
import numpy as np
import pandas as pd
import time
import os
import json
from collections import defaultdict
from trainer import fsMTS_train, enet_train, standardize, center_and_allstd, my_ard_train, my_lasso_train, bayesFS, my_enet_train, lasso_train, ard_regression_train
from attribution import gradients_, gradients_x_inputs_, deeplift_linear_
from utils import shift, is_all_zeros_or_nans
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
TIME_UNIT = 'min'
METRIC_TIME_COL = 'event_date'
SQL_TIME_COL = 'batch_time'


def get_tenant_data(allmetrics, allsqls, cluster, tenant_name, occur_time, before=60, after=10):
    """
    get tenant metrics and its SQL metrics
    @param allmetrics: all tenant metrics
    @param allsqls:    all sql metrics
    @param cluster:   abnormal cluster
    @param tenant_name: abnormal tenant name
    @param occur_time:  time when anomaly occurs T
    @param before:      how long we get before  T- before
    @param after:       how long we get after  T + after
    @return: tenant metrics DataFrame and tenant sql metrics DataFrame
    """
    # get start time and end time
    starttime = occur_time.ceil(TIME_UNIT) - pd.Timedelta(str(before) + TIME_UNIT)
    endtime = occur_time.ceil(TIME_UNIT) + pd.Timedelta(str(after) + TIME_UNIT)
    # print("time range: {} - {}".format(starttime, endtime))
    tenant_metrics = allmetrics.loc[(allmetrics.cluster == cluster) & (allmetrics.tenant_name == tenant_name)]
    tenant_metrics = tenant_metrics[(tenant_metrics[METRIC_TIME_COL] >= starttime) & (tenant_metrics[METRIC_TIME_COL] <= endtime)]
    # get new DataFrame， sort tenant_metrics by time to get time series
    tenant_metrics = tenant_metrics.sort_values(METRIC_TIME_COL)
    tenant_sqls = allsqls.loc[(allsqls.cluster == cluster) & (allsqls.tenant_name == tenant_name)]
    # get new DataFrame of sqls metrics, sorted by time to get time series
    tenant_sqls = tenant_sqls.sort_values(SQL_TIME_COL)
    return tenant_metrics, tenant_sqls


def prepare_train_data_v2(tenant_metrics, tenant_sqls, tenant_metric_names, required_sql_metrics, required_sql_types):
    """
    prepare train data， each pair (y:tenant_metric, X: sql_metric) , X columns are sql metrics of sql_ids, y is tenant metric)
    @param tenant_metrics:
    @param tenant_sqls:
    @param tenant_metric_names:
    @param required_sql_metrics:
    @param required_sql_types:
    @return:
    """
    tenant_metrics = tenant_metrics.rename(columns={'logical_reads': 'lr'})
    # train_y is a new dataframe including all tenant_metrics
    train_y = tenant_metrics[[METRIC_TIME_COL] + tenant_metric_names].reset_index(drop=True)
    train_x = defaultdict(dict)
    for tenant_metric_name in tenant_metric_names:
        # filter sql types
        subsql = tenant_sqls
        if tenant_metric_name in required_sql_types:
            sql_types = required_sql_types[tenant_metric_name]
            subsql = subsql[subsql['sql_type'].isin(sql_types)].copy()
        # keep top sqls
        sql_ids = subsql.groupby('sql_id')[['logical_reads']].max().sort_values('logical_reads', ascending=False).index.tolist()[:100]
        print("total num of required top sqlids", len(sql_ids))

        # prepare each pair of (X：pd.Dataframe， y：pd.Series)
        sql_metric_names = required_sql_metrics[tenant_metric_name]
        for sql_metric_name in sql_metric_names:
            # load same metric of different sql_ids, sql_metric: cpu_time, logical_reads etc.
            tmp = pd.DataFrame()
            for sql_id in sql_ids:

                onesql = subsql[subsql.sql_id == sql_id]
                if len(onesql) == 0:
                    continue
                # align X, y by timestamp
                data = train_y[['event_date']].merge(onesql, how='left', left_on='event_date', right_on='batch_time')
                # ignore empty sqls
                if len(data) == 0 or is_all_zeros_or_nans(data[sql_metric_name]):
                    continue
                tmp[sql_id] = data[sql_metric_name]

            train_x[tenant_metric_name][sql_metric_name] = tmp
            train_y['timestamp'] = train_y['event_date'].astype(int) // 1000000 - 8 * 3600 * 1000000
            print("{}, {} shape(n_times, n_sqls): {}".format(tenant_metric_name, sql_metric_name, tmp.shape))

    return train_x, train_y


def preprocessing(X, y, before, impute=True, leftshift=True):
    """
    impute， remove all zeros or nan sql_id, add a shift time series for each sql_id
    @param impute:
    @param X:
    @param y:
    @param before: index of anomaly occurs
    @return:
    """
    if len(X) == 0:
        return X, y
    if before > len(X) - 1:
        before = len(X) - 1
    if impute:
        # X.loc[before:, :] = X.loc[before:, :].interpolate()
        X = X.fillna(0.)
        y = y.interpolate()
        y = y.ffill()
        y = y.bfill()
    # X_allstd = X / np.nanstd(X.values)
    drops = []
    for sql_id in X.columns.tolist():
        if is_all_zeros_or_nans(X.loc[before - 1:before + 1, sql_id]):  # or X_allstd.loc[:, sql_id].std() < 1e-1:
            drops.append(sql_id)
    X.drop(drops, axis=1, inplace=True)
    if leftshift:
        for sql_id in X.columns.tolist():
            X[sql_id + '_ls1'] = shift(X[sql_id].values, -1, 0.)
    
    return X, y


def parse_detail_result(res) -> list:
    """
    remove duplicates with same sql_id
    @param res:
    @return:
    """
    finres = []
    for tenant_metric_name, sql_res in res.items():
        for sql_metric_name, single_res in sql_res.items():
            for col in single_res:
                if col in finres or col.replace('_ls1', '') in finres:
                    continue
                else:
                    finres.append(col.replace('_ls1', ''))
    return finres


def hit_details(res, truth, hits):
    """
    check accuracy details, hit by which pair sub-model
    @param res:
    @param truth:
    @param hits:
    @return:
    """
    # check and record results hit or miss and details
    # print(res)
    for tenant_metric_name, sql_res in res.items():
        for sql_metric_name, single_res in sql_res.items():
            single_set = set([sid.replace('_ls1', '') for sid in single_res])
            if truth in single_res or truth + '_ls1' in single_res:
                for k, it in enumerate(single_res):
                    if it == truth or it == truth + '_ls1':
                        k = k + 1
                        break
                hits[tenant_metric_name + ' ' + sql_metric_name] += 1
                print("\thit by {}, {}, hit {}th in {}".format(tenant_metric_name, sql_metric_name, k, len(single_set)))
            else:
                print("\tNOT hit by {}, {} in {}".format(tenant_metric_name, sql_metric_name, len(single_set)))
    return hits

@ignore_warnings(category=ConvergenceWarning)
def evaluate(train_x, train_y, tenant_metric_names, thd, before, after, trainer, attribution_method, impute=True, positive=True, normalize=None, num_out=3, leftshift=True):
    """
    evaluate one case
    @param train_x:
    @param train_y:
    @param tenant_metric_names:
    @param thd:
    @param before:
    @param after:
    @param trainer:
    @param attribution_method:
    @param impute:
    @param positive:
    @param normalize:
    @param num_out:
    @return:
    """
    # training for each pair of tenant_metric and sql_metric, res is Map<tenant_metric_name, <sql_metric_name, result_list>>
    res = defaultdict(dict)
    non_fit_num = 0
    for tenant_metric_name in tenant_metric_names:
        if len(np.nonzero(train_y[tenant_metric_name].values)[0]) <= 1 or train_y[tenant_metric_name].isna().sum() == len(train_y):
            print('No enough data in {}'.format(tenant_metric_name))
            non_fit_num += 1
            continue
        zero_x_num = 0
        for sql_metric_name, X in train_x[tenant_metric_name].items():
            X, y = preprocessing(X, train_y[tenant_metric_name], before=before, impute=impute, leftshift=leftshift)
            n, p = X.shape
            print("shape:", (n, p))
            if n > 0 and p > 0:
                if trainer == 'fsMTS':
                    model = fsMTS_train(X, y, normalize)

                elif trainer == 'bayesFS':
                    model = bayesFS(X, y, positive, normalize=normalize)
                elif trainer == 'myard':
                    model = my_ard_train(X, y, positive, normalize)
                elif trainer == 'mylasso':
                    model = my_lasso_train(X, y, positive, normalize)
                elif trainer == 'myenet':
                    model = my_enet_train(X, y, positive, normalize)
                elif trainer == 'ard':
                    model = ard_regression_train(X, y, normalize)
                elif trainer == 'enet':
                    model = enet_train(X, y, normalize)
                elif trainer == 'lasso':
                    model = lasso_train(X, y, normalize)
                else:
                    raise ValueError

                if attribution_method == 'gradients':
                    sorted_res = gradients_(model, X, thd, reverse=True)
                elif attribution_method == 'gradients_x_inputs':
                    sorted_res = gradients_x_inputs_(model, X, before, thd)
                elif attribution_method == 'deeplift_linear':
                    sorted_res = deeplift_linear_(model, X, before, thd)
                else:
                    raise ValueError
                if sorted_res:
                    res[tenant_metric_name][sql_metric_name] = sorted_res[:min(num_out, len(sorted_res))]
            else:
                zero_x_num += 1
        if zero_x_num == len(train_x[tenant_metric_name]):
            non_fit_num += 1
    return res, non_fit_num


# evaluate_all
def evaluate_all(allmetrics, allsqls, clusters, tenants, occur_times, labels, thd=0, before=60, after=10,
                 trainer='enet', attribution_method='gradients', impute=True, positive=False, normalize=None, leftshit=True):
    """
    evaluate all cases
    @param allmetrics:
    @param allsqls:
    @param clusters:
    @param tenants:
    @param occur_times:
    @param labels:
    @param thd:
    @param before:
    @param after:
    @param trainer:
    @param attribution_method:
    @param impute:
    @param positive:
    @param normalize:
    @return:
    """
    # configs
    tenant_metric_names = ['sql_select_rt', 'lr']
    required_sql_metrics = {
        'sql_select_rt': ['cpu_time', ],
        'cpu_usage_max': ['total_wait_time'],
        'lr': ['logical_reads'],
    }
    required_sql_types = {  # 'sql_select_rt': [1, 2],
    }

    # accumulator
    hit_num, miss_num = 0, 0
    hits = defaultdict(int)
    recommand_sum = 0
    time_sum = 0

    for c, (cluster, tenant_name, ts, truth) in enumerate(zip(clusters, tenants, occur_times, labels)):
        print(cluster, tenant_name, ts, truth)
        t0 = time.time()
        # get case data
        tenant_metrics, tenant_sqls = get_tenant_data(allmetrics, allsqls, cluster, tenant_name, ts, before=before, after=after)

        # prepare X and Y

        train_x, train_y = prepare_train_data_v2(tenant_metrics, tenant_sqls, tenant_metric_names, required_sql_metrics,
                                                 required_sql_types)

        res, non_fit_num = evaluate(train_x, train_y, tenant_metric_names, thd, before, after, trainer, attribution_method, impute, positive, normalize, 3, leftshit)

        hits = hit_details(res, truth, hits)

        # parse detail result 
        finres = parse_detail_result(res)
        if truth in finres:
            print("hit in {}".format(len(finres)))
            hit_num += 1
            recommand_sum += len(finres)
        elif non_fit_num != len(tenant_metric_names):
            print("miss in {}".format(len(finres)))
            print(finres)
            miss_num += 1
            recommand_sum += len(finres)
        print("time: {}".format(time.time() - t0))
        time_sum += time.time() - t0
        print("current fit precision: {} / {} = {}".format(hit_num, hit_num + miss_num, hit_num / (hit_num + miss_num)))
        if hit_num != 0:
            print("current average recommanded num of sql: {} / {} = {}".format(recommand_sum, hit_num+miss_num,
                                                                                recommand_sum / (hit_num+miss_num)))
        print(finres)
        print("\n")

    # print("precision: {} / {} = {}".format(hit_num, len(labels), hit_num / len(labels)))
    print("Available cases precision: {} / {} = {}".format(hit_num, hit_num + miss_num, hit_num / (hit_num + miss_num)))
    print("average recommanded num of sql: {} / {} = {}".format(recommand_sum, hit_num+miss_num, recommand_sum / (hit_num + miss_num)))
    print("average time of computing: {} / {} = {}".format(time_sum, len(labels), time_sum / len(labels)))
    return hit_num, miss_num, len(labels), hits, recommand_sum / (hit_num + miss_num), time_sum / len(labels)
