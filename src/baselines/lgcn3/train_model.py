import random as rd
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import test_model
from model_LGCN_afd_tri import model_LGCN_afd_tri  # LGCN AFD tri-partite version
from model_LGCN_tri import model_LGCN_tri  # LGCN tri-partite version
from model_LightGCN_tri import model_LightGCN_tri  # 표준 LightGCN tri-partite version (Issue #30)
from print_save import save_value
from test_model import test_model, test_model_store
from tqdm import tqdm


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
    F1_df_train = pd.DataFrame(columns=TOP_K) # to log the training loss's changes
    NDCG_df = pd.DataFrame(columns=TOP_K)
    t1 = time.perf_counter()

    # training loops
    with tqdm(total=N_EPOCH) as pbar:
        for epoch in range(N_EPOCH):
            for batch_num in range(len(batches) - 1):
                train_batch_data = []
                for sample in range(batches[batch_num], batches[batch_num + 1]):
                    (user, pos_item) = train_data_interaction[sample]
                    sample_num = 0
                    while sample_num < (SAMPLE_RATE if (MODEL in ['LGCN', 'LGCN_tri', 'LGCN_AFD_tri', 'LGCN_AFD']) else 1):
                        neg_item = int(rd.uniform(0, item_num)) # sample random exclusive items as the negative
                        if neg_item not in train_data[user]:
                            sample_num += 1
                            train_batch_data.append([user, pos_item, neg_item])
                train_batch_data = np.array(train_batch_data)
                _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2], model.keep_prob: KEEP_PORB if MODEL in ['LGCN', 'LGCN_tri', 'LGCN_AFD_tri', 'LGCN_AFD'] else 1})
            ## test the model each epoch
            F1, NDCG = test_model(sess, model, para_test)
            F1_max = max(F1_max, F1[0])
            # F1_train, NDCG_train = test_model_train(sess, model, para_test)
            ## print performance
            # print_value([epoch + 1, loss, F1_max, F1, NDCG])
            # if epoch % 10 == 0: print('%.5f' % (F1_max), end = ' ', flush = True)
            pbar.set_description(f"F1_max: {F1_max :2f}")
            pbar.update(1)
            ## save performance
            F1_df.loc[epoch + 1] = F1
            NDCG_df.loc[epoch + 1] = NDCG
            # F1_df_train.loc[epoch + 1] = F1_train
            # save_value([[F1_df, 'F1'], [F1_df_train, 'F1_train'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
            save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
            if loss > 10 ** 10: break
    t2 = time.perf_counter()
    print('time cost:', (t2 - t1) / 200)

    if results_save_path:
        print('Saving results...')
        test_model_store(sess, model, para_test, results_save_path)
        print('Well saved.')

    return F1_max
