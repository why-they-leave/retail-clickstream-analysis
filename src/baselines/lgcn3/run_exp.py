# builtin
from datetime import datetime

import params

# internal
import read_data

# external
# import matplotlib.pyplot as plt
import tensorflow as tf
import train_model
from params import AFD_ALPHA, BATCH_SIZE, DATASET, LAMDA, LAYER, LR, MODEL, OPTIMIZATION

tf.compat.v1.disable_eager_execution() # to disable the eager mode


res_dir = "./exp_res/"
target_dir = res_dir + f'{MODEL}/'
# check
# assert params.all_para[2] == 'LightRGCN'
print(params.all_para)
# read
read_data_res_tri = read_data.read_all_data_tri(params.all_para, approximate=False)
# run
today = datetime.today()
formatted_date = today.strftime('%Y%m%d')
for i in range(3):
    if 'AFD' in MODEL:
        exsl_path = f'{DATASET}_{MODEL}_{formatted_date}_{LR}_{LAMDA}_{LAYER}_{BATCH_SIZE}_{OPTIMIZATION}_{AFD_ALPHA}_{i}.xlsx'
    else:
        exsl_path = f'{DATASET}_{MODEL}_{formatted_date}_{LR}_{LAMDA}_{LAYER}_{BATCH_SIZE}_{OPTIMIZATION}_{i}.xlsx'
    # F1_max = train_model.train_model(params.all_para[:26], read_data_res_tri, target_dir + exsl_path, '')
    F1_max = train_model.train_model(params.all_para, read_data_res_tri, target_dir + exsl_path, '')
