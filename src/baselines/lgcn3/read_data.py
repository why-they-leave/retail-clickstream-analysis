import json
import random as rd

import numpy as np
from dense2sparse import propagation_matrix, propagation_matrix_tri


def read_data_tri(path_u2t, path_t2p, path_u2p,):
    # read files
    with open(path_u2t, 'r') as f:
        tri_graph_uidx2tidx_train = json.load(f)
    with open(path_t2p, 'r') as f:
        tri_graph_tidx2pidx = json.load(f)
    with open(path_u2p, 'r') as f:
        tri_graph_uidx2pidx = json.load(f)

    # trans key to int
    tri_graph_uidx2tidx_train = {int(k):v for k,v in tri_graph_uidx2tidx_train.items()}
    tri_graph_tidx2pidx = {int(k):v for k,v in tri_graph_tidx2pidx.items()}
    tri_graph_uidx2pidx = {int(k):v for k,v in tri_graph_uidx2pidx.items()}

    # print(len(tri_graph_uidx2tidx_train), len(tri_graph_tidx2pidx), len(tri_graph_uidx2pidx)) # debug

    # make data
    user_num = len(tri_graph_uidx2tidx_train)
    item_num = len(tri_graph_tidx2pidx)

    u2t_data = [tri_graph_uidx2tidx_train[uidx] for uidx in range(user_num)] # [[tidx,]]
    u2p_data = [tri_graph_uidx2pidx.get(uidx, []) for uidx in range(user_num)] # [[pidx,]]
    # notice: u2p is not available for some users
    t2p_data = [tri_graph_tidx2pidx[tidx] for tidx in range(item_num)] # [[pidx,]]

    # make interactions
    interactions_u2t = []
    interactions_u2p = []
    interactions_t2p = []
    for user in range(user_num): # user: uidx
        for item in u2t_data[user]: # item: tidx
            interactions_u2t.append((user, item))
        for persona in u2p_data[user]:
            interactions_u2p.append((user, persona))
    for item in range(item_num):
        for entry in t2p_data[item]:
            # Issue #29: entry는 persona_id(기존) 또는 [persona_id, lift_weight](신규, item-segment
            # lift 가중치 지원)를 모두 지원한다.
            if isinstance(entry, (list, tuple)):
                persona, weight = entry[0], entry[1]
                interactions_t2p.append((item, persona, weight))
            else:
                interactions_t2p.append((item, entry))

    # shuffle for randomness
    rd.shuffle(interactions_u2t) # [TODO] uncontrolled shuffle
    rd.shuffle(interactions_u2p)
    rd.shuffle(interactions_t2p)

    return(u2t_data, interactions_u2t, interactions_u2p, interactions_t2p, user_num, item_num)

def read_bases(path, fre_u, fre_v):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_v = np.array(feat_v)[:, 0: fre_v].astype(np.float32)
    return [feat_u, feat_v]

def read_bases1(path, fre, _if_norm = False):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    if _if_norm:
        for i in range(len(bases)):
            bases[i] = bases[i]/np.sqrt(np.dot(bases[i], bases[i]))
    return np.array(bases)[:, 0: fre].astype(np.float32)

def read_all_data_tri(all_para, approximate=False):
    # approximate=True will read the approximated ver of tri_graph_uidx2pidx
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    # data/ 최상위가 아니라 data/processed/를 가리키도록 수정 (Issue #29 — 프로젝트 데이터 폴더 관례 준수)
    DIR = '../../../data/processed/'

    ## Load data
    ## load training data
    print('Reading data...')
    path_u2t = DIR + 'tri_graph_uidx2tidx_train.json'
    path_t2p = DIR + 'tri_graph_tidx2pidx.json'
    if approximate:
        # path_u2p = DIR + 'tri_graph_uidx2pidx_approach.json'
        path_u2p = DIR + 'tri_graph_uidx2pidx_app_e_0.1.json'
    else:
        path_u2p = DIR + 'tri_graph_uidx2pidx.json'
    # [train_data, train_data_interaction, user_num, item_num] = read_data_tri(path_u2t, path_t2p)
    [train_data, train_data_interaction, interactions_u2p, interactions_t2p, user_num, item_num] = read_data_tri(path_u2t, path_t2p, path_u2p)

    ## load test data
    test_vali_path = DIR + 'tri_graph_uidx2tidx_valid.json' if TEST_VALIDATION == 'Validation' else DIR + 'tri_graph_uidx2tidx_test.json'
    test_data = read_data_tri(test_vali_path, path_t2p, path_u2p)[0]

    if DATASET == 'MBA':
        persona_num = 20 # 20, for MBA
    elif DATASET == 'Instacart':
        persona_num = 51 # 51, fixed for Instacart
    elif DATASET == 'Instacart_full':
        persona_num = 51

    # graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings

    if MODEL in ['LGCN', 'LGCN_tri']:
        if GRAPH_CONV == '1D':
            if not approximate:
                # normal case
                graph_embeddings_1d_path = DIR + 'graph_embeddings_1d_tri.json' if MODEL == 'LGCN_tri' else  DIR + 'graph_embeddings_1d.json'   # 1d graph embeddings
            else:
                # approximated case
                graph_embeddings_1d_path = DIR + 'graph_embeddings_1d_tri_approach.json' if MODEL == 'LGCN_tri' else  DIR + 'graph_embeddings_1d.json'   # 1d graph embeddings

            print(f'Reading graph_embeddings_1d from path: {graph_embeddings_1d_path}')
            ## load pre-trained transform bases for LCFN and SGNN
            graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY) # same as LGCN
        else: assert False, f'Not supported: {MODEL}, {GRAPH_CONV}'
    elif MODEL in ['LightGCN', 'LightGCN_tri', 'NGCF', 'GCMC']:
        if MODEL == 'LightGCN_tri':
            graph_tri = [train_data_interaction, interactions_u2p, interactions_t2p]
            sparse_propagation_matrix = propagation_matrix_tri(graph_tri, user_num, item_num, persona_num, 'sym_norm')
        elif MODEL == 'LightGCN':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
        elif MODEL == 'NGCF':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
        elif MODEL == 'GCMC':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'left_norm')

    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False
    else:
        pre_train_feature = [0, 0]

    print('Data all read successfully!')
    return train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN
    # 0:train_data, 1:train_data_interaction, 2:user_num, 3:item_num, 4:persona_num, 5:test_data,
    # 6:pre_train_feature, 7:hypergraph_embeddings, 8:graph_embeddings, 9:propagation_embeddings,
    # 10:sparse_propagation_matrix, 11:IF_PRETRAIN
