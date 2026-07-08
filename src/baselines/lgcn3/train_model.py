import random as rd
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import test_model
from model_LGCN_afd_tri import model_LGCN_afd_tri  # LGCN AFD tri-partite version
from model_LGCN_tri import model_LGCN_tri  # LGCN tri-partite version
from model_LightGCN_tri import model_LightGCN_tri  # 표준 LightGCN tri-partite version (Issue #30)
from print_save import save_value
from test_model import test_model, test_model_store


def train_model(para, data, path_excel, results_save_path=''):
    ## data and hyperparameters
    [train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, _] = data
    # PROP_DIM, PROP_EMB, IF_NORM, AFD_ALPHA
    [_, _, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN, _, TOP_K] = para[0:13]
    # if MODEL == 'LGCN' or MODEL == 'LGCN_tri' or MODEL == 'LightGCN_tri' or MODEL == 'LightRGCN' or MODEL == 'LightFCN_AFD':
    if MODEL in ['LGCN', 'LGCN_tri', 'LightGCN_tri', 'LightRGCN', 'LightGCN_AFD', 'LightGCN_AFD_tri', 'LGCN_AFD_tri', 'LGCN_AFD']:
        [_, _, _, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING, _, _, _, AFD_ALPHA] = para[13:]
    if MODEL == 'SGNN': [_, PROP_EMB, _] = para[13:]
    para_test = [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH]
    ## Define the model
    if MODEL == 'MF': model = model_MF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA)
    if MODEL == 'NCF': model = model_NCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'GCMC': model = model_GCMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'NGCF': model = model_NGCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'SCF': model = model_SCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'CGMC': model = model_CGMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LightGCN': model = model_LightGCN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LCFN': model = model_LCFN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, graph_embeddings=hypergraph_embeddings)
    if MODEL == 'LGCN': model = model_LGCN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    if MODEL == 'SGNN': model = model_SGNN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, propagation_embeddings=propagation_embeddings, if_pretrain=IF_PRETRAIN, prop_emb=PROP_EMB)
    if MODEL == 'LGCN_tri': model = model_LGCN_tri(n_users=user_num, n_items=item_num, n_personas=persona_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    # 표준 LightGCN이라 pre_train_latent_factor/if_pretrain 등 spectral 전용 인자가 없음 (Issue #30)
    if MODEL == 'LightGCN_tri': model = model_LightGCN_tri(n_users=user_num, n_items=item_num, n_personas=persona_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, sparse_graph=sparse_propagation_matrix, optimization=OPTIMIZATION)
    if MODEL == 'LightRGCN': model = model_LightRGCN(layer=LAYER, n_users=user_num, n_items=item_num, n_personas=persona_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LightGCN_AFD': model = model_LightGCN_afd(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix, afd_alpha=AFD_ALPHA)
    if MODEL == 'LightGCN_AFD_tri': model = model_LightGCN_afd_tri(layer=LAYER, n_users=user_num, n_items=item_num, n_personas=persona_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix, optimization=OPTIMIZATION, afd_alpha=AFD_ALPHA)
    if MODEL == 'LGCN_AFD_tri': model = model_LGCN_afd_tri(n_users=user_num, n_items=item_num, n_personas=persona_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING, afd_alpha=AFD_ALPHA)
    if MODEL == 'LGCN_AFD': model = model_LGCN_afd(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING, afd_alpha=AFD_ALPHA)
    # return model

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    NDCG_df = pd.DataFrame(columns=TOP_K)
    # Issue #30: epoch당 loss 기록(기존엔 print_value 호출이 주석 처리돼 있어 loss를
    # 전혀 볼 수 없었음) — 학습이 실제로 수렴하는지 확인할 유일한 신호라 켜둔다.
    loss_df = pd.DataFrame(columns=["loss"])
    # Issue #30: 매 epoch 무작위 512명을 새로 뽑으면(기존 동작) user_num 대비 정답 있는
    # 유저 비율이 낮아 유효 평가 인원이 매번 다르고 극소수라 F1 곡선이 순수 노이즈에
    # 가까웠다. 정답이 있는 유저 전체로 고정해 epoch 간 비교가 가능하게 한다.
    fixed_test_batch = [u for u in range(user_num) if len(test_data[u]) > 0]
    t1 = time.perf_counter()

    # training loops
    with tqdm(total=N_EPOCH) as pbar:
        for epoch in range(N_EPOCH):
            for batch_num in range(len(batches) - 1):
                train_batch_data = []
                for sample in range(batches[batch_num], batches[batch_num + 1]):
                    (user, pos_item) = train_data_interaction[sample]
                    sample_num = 0
                    # Issue #30: LightGCN_tri도 SAMPLE_RATE(양성당 음성 개수)를 쓰도록 목록에 추가.
                    # keep_prob 목록(아래 줄)은 그대로 둔다 — LightGCN_tri는 dropout이 없어서 항상 1이어야 함.
                    while sample_num < (SAMPLE_RATE if (MODEL in ['LGCN', 'LGCN_tri', 'LGCN_AFD_tri', 'LGCN_AFD', 'LightGCN_tri']) else 1):
                        neg_item = int(rd.uniform(0, item_num)) # sample random exclusive items as the negative
                        if neg_item not in train_data[user]:
                            sample_num += 1
                            train_batch_data.append([user, pos_item, neg_item])
                train_batch_data = np.array(train_batch_data)
                _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2], model.keep_prob: KEEP_PORB if MODEL in ['LGCN', 'LGCN_tri', 'LGCN_AFD_tri', 'LGCN_AFD'] else 1})
            ## test the model each epoch (고정된 유저 집합으로 평가 — 위 #30 참고)
            F1, NDCG = test_model(sess, model, para_test, fixed_test_batch=fixed_test_batch)
            F1_max = max(F1_max, F1[0])
            pbar.set_description(f"F1_max: {F1_max:.5f} loss: {loss:.4f}")
            pbar.update(1)
            ## save performance
            F1_df.loc[epoch + 1] = F1
            NDCG_df.loc[epoch + 1] = NDCG
            loss_df.loc[epoch + 1] = [float(loss)]
            save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG'], [loss_df, 'Loss']], path_excel, first_sheet=False)
            if loss > 10 ** 10: break
    t2 = time.perf_counter()
    print('time cost:', (t2 - t1) / 200)

    if results_save_path:
        print('Saving results...')
        test_model_store(sess, model, para_test, results_save_path)
        print('Well saved.')

    # Issue #30: run_lightgcn.py가 학습 직후 전체 유저 추천(top_items/top_scores)을
    # 뽑으려면 학습된 그래프가 살아있는 sess/model이 필요하다. sess는 with 블록 없이
    # 열려있어 함수 밖에서도 안전하게 쓸 수 있다 (기존 동작 그대로, 반환값만 추가).
    return F1_max, sess, model
