### BALANCE: Bayesian Linear Attribution for Root Cause Localization

This is the implementations of the SIGMOD23 paper [BALANCE: Bayesian Linear Attribution for Root Cause Localization](https://arxiv.org/pdf/2301.13572.pdf) 

The algorithm code are within `BMFS.py, attribution.py.`
`trainer.py` is for training a case using different methods.

All experiments in paper are initialized in jupyter notebook demos.
`/bad_sql_experiments` contains the experiments data and code for "Bad SQL Localization".
Its data is in bad_sql_data.zip and demo code is in badsql_v3_local.ipynb

`/container_fault_experiments` contains the experiments data and code for "Container Fault Localization".
Its data is in container_fault_dataset.csv and label.json. And demo code is in container_fault.ipynb

`/exathlon_experiments` contains experiment code for "Exathlon Localizaiton". Original exathlon data could be downloaded from https://github.com/exathlonbenchmark/exathlon.
For easy usage, `cases.pickle` can be unzip from `cases.pickle.zip` and it is a preprocessed data from raw data of public dataset Exathlon including what we need in this paper. However exthalon dataset offers more other possibilities such as anomaly detection.
Our experiment is demonstrated in exathlon_rca.ipynb and instructions in the paper.

`/synthetic_experiments` contains demo code for generating synthetic data and tests.

### Citation

`@inproceedings{chen2023balance,
title={BALANCE: Bayesian Linear Attribution for Root Cause Localization},
author={Chen, Chaoyu and Yu, Hang and Lei, Zhichao and Li, Jianguo and Ren, Shaokang and Zhang, Tingkai and Hu, Silin and Wang, Jianchao and Shi, Wenhui},
booktitle={Proceedings of the 2023 ACM SIGMOD International Conference on Management of Data},
year={2023}
}`

