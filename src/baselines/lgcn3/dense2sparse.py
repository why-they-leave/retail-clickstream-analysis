import numpy as np
import tensorflow as tf


def propagation_matrix(graph, user_num, item_num, norm):
    print('Constructing the sparse graph...')
    eps = 0.1 ** 10
    user_itemNum = np.zeros(user_num)
    item_userNum = np.zeros(item_num)
    for (user, item) in graph:
        user_itemNum[user] += 1
        item_userNum[item] += 1
    val, idx = [], []
    for (user, item) in graph:
        if norm == 'left_norm':
            val += [1 / max(user_itemNum[user], eps), 1 / max(item_userNum[item], eps)]
            idx += [[user, item + user_num], [item + user_num, user]]
        if norm == 'sym_norm':
            val += [1 / (max(np.sqrt(user_itemNum[user] * item_userNum[item]), eps))] * 2
            idx += [[user, item + user_num], [item + user_num, user]]
    return tf.SparseTensor(indices=idx, values=val, dense_shape=[user_num + item_num, user_num + item_num])

def propagation_matrix_tri(graph, user_num, item_num, persona_num, norm):
    print('Constructing the tripartite sparse graph...')
    [uidx2tidx_graph, uidx2pidx_graph, tidx2pidx_graph] = graph # parse the 3 subgraphs

    eps = 0.1 ** 10
    user_degreeNum = np.zeros(user_num)
    item_degreeNum = np.zeros(item_num)
    persona_degreeNum = np.zeros(persona_num)

    for (user, item) in uidx2tidx_graph:
        user_degreeNum[user] += 1
        item_degreeNum[item] += 1

    for (user, persona) in uidx2pidx_graph:
        user_degreeNum[user] += 1
        persona_degreeNum[persona] += 1

    for (item, persona) in tidx2pidx_graph:
        item_degreeNum[item] += 1
        persona_degreeNum[persona] += 1

    if norm == 'sym_norm':
        val, idx = [], []
        for (user, item) in uidx2tidx_graph:
            val += [1 / (max(np.sqrt(user_degreeNum[user] * item_degreeNum[item]), eps))] * 2
            idx += [[user, item + user_num], [item + user_num, user]]
        for (user, persona) in uidx2pidx_graph:
            val += [1 / (max(np.sqrt(user_degreeNum[user] * persona_degreeNum[persona]), eps))] * 2
            idx += [[user, persona + user_num + item_num], [persona + user_num + item_num, user]]
        for (item, persona) in tidx2pidx_graph:
            val += [1 / (max(np.sqrt(item_degreeNum[item] * persona_degreeNum[persona]), eps))] * 2
            idx += [[item + user_num, persona + user_num + item_num], [persona + user_num + item_num, item + user_num]]
        return tf.SparseTensor(indices=idx, values=val, dense_shape=[user_num + item_num + persona_num, user_num + item_num + persona_num])

    else: assert False, f'Not supported: {norm}'
