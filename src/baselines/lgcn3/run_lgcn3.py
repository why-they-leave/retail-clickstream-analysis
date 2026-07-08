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
# 참고: 이 스크립트는 LGCN_tri(사전학습 frequency embedding 필요) 전용이다.
# #30에서 이 경로 대신 LightGCN_tri + run_lightgcn.py로 진행하기로 결정했다
# (docs/LIGHTGCN_TRI_MODEL_DESIGN.md의 A/B안 비교 참고) — 이 파일은 그대로 남겨두되
# train_model()의 반환값 변경(F1_max, sess, model)에만 맞춰 언패킹을 수정한다.
for exsl_path in [f'MBA_{params.all_para[2]}_{params.all_para[3]}_{params.all_para[4]}.xlsx',]:
    F1_max, _, _ = train_model.train_model(params.all_para[:26], read_data_res, tri_lgcn_dir + exsl_path, '')
print(f'F1_max:{F1_max}')
