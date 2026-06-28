import numpy as np
import params
import read_data
import train_model

res_dir = "./exp_res/"
tri_lgcn_dir = res_dir

# reads data
assert params.all_para[2] == 'LGCN_tri'
read_data_res = read_data.read_all_data_tri(params.all_para, approximate=False)
print(f'pretrained embedding size: {read_data_res[8].shape}')
print(f'interaction number: {np.sum([len(v) for v in read_data_res[0]])}')
print(f'params: {params.all_para}')

# train and test
for exsl_path in [f'MBA_{params.all_para[2]}_{params.all_para[3]}_{params.all_para[4]}.xlsx',]:
    F1_max = train_model.train_model(params.all_para[:26], read_data_res, tri_lgcn_dir + exsl_path, '')
print(f'F1_max:{F1_max}')
