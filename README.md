This is the implementations of paper BALANCE https://arxiv.org/pdf/2301.13572.pdf.

The algorithm code are within BMFS.py, attribution.py.
trainer.py is for training a case using different methods.

All experiments in paper are initialized in jupyter notebook demos.
/bad_sql_experiments contains the experiments data and code for "Bad SQL Localization".
Its data is in bad_sql_data.zip and demo code is in badsql_v3_local.ipynb

/container_fault_experiments contains the experiments data and code for "Container Fault Localization".
Its data is in container_fault_dataset.csv and label.json. And demo code is in container_fault.ipynb

/exathlon_experiments contains experiment code for "Exathlon Localizaiton".
cases.pickle can be unzip from cases.pickle.zip it is a prepared data from raw data of public dataset Exathlon(https://github.com/exathlonbenchmark/exathlon). 
You can just follow exathlon_rca.ipynb and paper.

/synthetic_experiments contains demo code for generating synthetic data and tests.

