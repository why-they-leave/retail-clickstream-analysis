import json
import random as rd

import numpy as np
from numpy import *

from evaluation import evaluation_F1, evaluation_NDCG


def test_one_user(user, top_item, para_test_one_user):
    [test_data, TOP_K] = para_test_one_user
    k_num = len(TOP_K)
    f1 = np.zeros(k_num)
    ndcg = np.zeros(k_num)
    top_item = top_item.tolist()  ## make testing fatser
    for i in range(k_num):
        f1[i] = evaluation_F1(top_item, TOP_K[i], test_data[user])
        ndcg[i] = evaluation_NDCG(top_item, TOP_K[i], test_data[user])
    return f1, ndcg

def test_model(sess, model, para_test):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    para_test_one_user = [test_data, TOP_K]
    ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
    ## For some heavy models (e.g., NCF and LGCN with MLP as the predictor), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
    user_top_items = np.zeros((TEST_USER_BATCH, max(TOP_K))).astype(dtype=int32)
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num)) # split the test batch in 100 mini batches
    mini_batch_list.append(TEST_USER_BATCH)
    score_min = -10 ** 5
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1] # user range for this mini-batch
        user_batch = test_batch[u1: u2] # mini batch of users
        items_in_train_data = np.zeros((u2 - u1, item_num)) # matrix: mini_batch_size * item_name
        for u_index, user in enumerate(user_batch):
            for item in train_data[user]:
                # items_in_train_data: score with a very large negative value, to rank the items bought by the user the training set to the tail, avoiding recommend them
                items_in_train_data[u_index, item] = score_min
        user_top_items_batch = sess.run(model.top_items, feed_dict={model.users: user_batch, model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: max(TOP_K)})
        user_top_items[u1: u2] = user_top_items_batch
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user(user, user_top_items[u_index], para_test_one_user))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    return F1, NDCG

def test_model_store(sess, model, para_test, save_path):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    para_test_one_user = [test_data, TOP_K]
    ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
    ## For some heavy models (e.g., NCF and LGCN with MLP as the predictor), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
    user_top_items = np.zeros((TEST_USER_BATCH, max(TOP_K))).astype(dtype=int32)
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num)) # split the test batch in 100 mini batches
    mini_batch_list.append(TEST_USER_BATCH)
    score_min = -10 ** 5
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1] # user range for this mini-batch
        user_batch = test_batch[u1: u2] # mini batch of users
        items_in_train_data = np.zeros((u2 - u1, item_num)) # matrix: mini_batch_size * item_name
        for u_index, user in enumerate(user_batch):
            for item in train_data[user]:
                # items_in_train_data: score with a very large negative value, to rank the items bought by the user the training set to the tail, avoiding recommend them
                items_in_train_data[u_index, item] = score_min
        user_top_items_batch = sess.run(model.top_items, feed_dict={model.users: user_batch, model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: max(TOP_K)})
        user_top_items[u1: u2] = user_top_items_batch
    results = {}
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            test_list = [int(e) for e in list(test_data[user])]
            pred_list = [int(e) for e in list(user_top_items[u_index][:100])]
            results[user] = [u_index, test_list, pred_list]
    with open(save_path, 'w') as file:
        json.dump(results, file)


def test_model_train(sess, model, para_test):
    [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH] = para_test
    para_test_one_user = [train_data, TOP_K] # changed
    ## Since Amazon is too large to calculate user_num*item_num interactions, we select TEST_USER_BATCH users to test the model.
    ## For some heavy models (e.g., NCF and LGCN with MLP as the predictor), calculating TEST_USER_BATCH*item_num interactions is still space-consuming, we split TEST_USER_BATCH users into mini batches further
    user_top_items = np.zeros((TEST_USER_BATCH, max(TOP_K))).astype(dtype=int32)
    test_batch = rd.sample(list(range(user_num)), TEST_USER_BATCH)
    mini_batch_num = 100
    mini_batch_list = list(range(0, TEST_USER_BATCH, mini_batch_num)) # split the test batch in 100 mini batches
    mini_batch_list.append(TEST_USER_BATCH)
    score_min = -10 ** 5
    for u in range(len(mini_batch_list) - 1):
        u1, u2 = mini_batch_list[u], mini_batch_list[u + 1] # user range for this mini-batch
        user_batch = test_batch[u1: u2] # mini batch of users
        items_in_train_data = np.zeros((u2 - u1, item_num)) # matrix: mini_batch_size * item_name
        # for u_index, user in enumerate(user_batch):
            # for item in train_data[user]:
            #     # items_in_train_data: score with a very large negative value, to rank the items bought by the user the training set to the tail, avoiding recommend them
            #     items_in_train_data[u_index, item] = score_min
        user_top_items_batch = sess.run(model.top_items, feed_dict={model.users: user_batch, model.keep_prob: 1, model.items_in_train_data: items_in_train_data, model.top_k: max(TOP_K)})
        user_top_items[u1: u2] = user_top_items_batch
    result = []
    for u_index, user in enumerate(test_batch):
        if len(test_data[user]) > 0:
            result.append(test_one_user(user, user_top_items[u_index], para_test_one_user))
    result = np.array(result)
    F1, NDCG = np.mean(np.array(result), axis=0)
    return F1, NDCG
