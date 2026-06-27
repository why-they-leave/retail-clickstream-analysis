import heapq
from copy import deepcopy
import sys
import importlib
import json
import random
from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

sys.path.append('../')
import random_walker.PageRank as pagerank

NTYP_U = 'unlabeled'
NTYP_L = 'labeled'
NTYP_V = 'item'
NTYP_P = 'persona'

class node:
    def __init__(self, id, typ, layer, in_nodes=[], out_nodes=[]):
        self.id = id
        self.typ = typ
        self.layer = layer
        self.in_nodes = set(in_nodes)
        self.out_nodes = set(out_nodes)

    def add_in_nodes(self, new_in_nodes):
        self.in_nodes.update(new_in_nodes)

    def add_out_nodes(self, new_out_nodes):
        self.out_nodes.update(new_out_nodes)

    def out_deg(self,):
        return len(self.out_nodes)

    def in_deg(self,):
        return len(self.in_nodes)

    def __str__(self,):
        return f'node id={self.id}, typ={self.typ}'

    def print_neighbors(self,):
        print('in neighbors:'+' '.join(n.id for n in self.in_nodes))
        print('out neighbors:'+' '.join(n.id for n in self.out_nodes))

class graph:
    def __init__(self, nodes):
        # nodes: []
        self.nodes = nodes
        self.u_nodes = [] # unlabeled
        self.l_nodes = [] # labeled
        self.v_nodes = [] # item
        self.p_nodes = []
        self.nodes_dict = {}
        self.layers = {}
        for n in self.nodes:
            # for trace
            self.nodes_dict[n.id] = n
            # for types
            if n.typ == NTYP_U: self.u_nodes.append(n)
            elif n.typ == NTYP_L: self.l_nodes.append(n)
            elif n.typ == NTYP_V: self.v_nodes.append(n)
            elif n.typ == NTYP_P: self.p_nodes.append(n)
            # for layers
            if self.layers.get(n.layer) is None:
                self.layers[n.layer] = []
            self.layers[n.layer].append(n)

    def __getitem__(self, nid):
        return self.nodes_dict.get(nid) # return None if not exists

def make_node(id, typ, layer, all_nodes, nodes_dict):
    new_node = node(id=id, typ=typ, layer=layer)
    all_nodes.append(new_node)
    nodes_dict[id] = new_node
    return new_node

def construct_graph(unlabeled_uidxs, labeled_uidxs, tri_graph_uidx2pidx, tri_graph_uidx2tidx_train):
    all_nodes = []
    nodes_dict = {}
    unlabeled_uidxs_set = set(unlabeled_uidxs)
    labeled_uidxs_set = set(labeled_uidxs)
    
    # for layer 3: unlabled users
    for uidx in unlabeled_uidxs:
        make_node('u'+str(uidx), NTYP_U, 3, all_nodes, nodes_dict)

    # for layer 1: labeled and unlabeled users
    for uidx in unlabeled_uidxs:
        make_node('u_'+str(uidx), NTYP_U, 1, all_nodes, nodes_dict)
    for uidx in labeled_uidxs:
        make_node('l'+str(uidx), NTYP_L, 1, all_nodes, nodes_dict)

    # for layer 0 (persona) and 0-1 edges
    for uidx in labeled_uidxs:
        lnode = nodes_dict['l'+str(uidx)]
        personas = tri_graph_uidx2pidx[uidx]
        update_pnodes = []
        for pidx in personas:
            pnid = 'p'+str(pidx)
            if nodes_dict.get(pnid) is None:
                make_node(pnid, NTYP_P, 0, all_nodes, nodes_dict)
            pnode = nodes_dict[pnid]
            pnode.add_in_nodes([lnode])
            update_pnodes.append(pnode)
        lnode.add_out_nodes(update_pnodes)

    # lastly, for layer 2 (items) and 1-2 edges and 2-3 edges
    for uidx, items in tri_graph_uidx2tidx_train.items():
        if uidx in unlabeled_uidxs_set: flag = 0 # 0 for unlabeled
        elif uidx in labeled_uidxs_set: flag = 1 # 1 for labeled
        else: flag = 2 # 2 for unrep

        if flag == 0:
            # unlabeled cases
            user_node1 = nodes_dict['u_'+str(uidx)]
            user_node2 = nodes_dict['u'+str(uidx)]
        elif flag == 1:
            # labeled case
            user_node1 = nodes_dict['l'+str(uidx)]
        else: # flag = 2
            continue # omit
        
        # update items
        update_tnodes = []
        for tidx in items:
            tnid = 't'+str(tidx)
            # make a new node if item does not exist
            if nodes_dict.get(tnid) is None:
                make_node(tnid, NTYP_V, 2, all_nodes, nodes_dict)
            item_node = nodes_dict[tnid]
            item_node.add_out_nodes([user_node1])
            if flag == 0:
                item_node.add_in_nodes([user_node2])
            update_tnodes.append(item_node)

        user_node1.add_in_nodes(update_tnodes)
        if flag == 0:
            user_node2.add_out_nodes(update_tnodes)
    
    return graph(all_nodes)

class max_heapq:
    # Assumption-1: no repeat node id (nid)
    def __init__(self,):
        self.data = [] # stores tuples in the form of [(value, nid),]
        self.entry_finder = {} # only traces the alive entries
        self.REMOVED = '<removed-task>'
    
    def push(self, nid, p):
        if self.entry_finder.get(nid) is not None:
            # update the original value: remove + repush
            self.remove(nid)
        entry = [-p, nid] # must be a list object
        self.entry_finder[nid] = entry # trace the new entry by nid
        heapq.heappush(self.data, entry)

    def pop(self,):
        while self.data:
            minus_p, nid = heapq.heappop(self.data)
            if nid != self.REMOVED:
                del self.entry_finder[nid]
                return -minus_p, nid
        print('[err]max_heapq: pop from an empty queue')
        return None # if no more elements in the Q

    def max(self,):
        while self.data:
            minus_p, nid = self.data[0]
            if nid == self.REMOVED:
                heapq.heappop(self.data)
            else:
                return -minus_p, nid
        return -1, None

    def remove(self, nid):
        rm_entry = self.entry_finder.pop(nid) # entry_finder no more trace the removed entry
        rm_entry[-1] = self.REMOVED # remove the nid to be the REMOVED signal

    def alive_num(self,):
        return len(self.entry_finder)

# # unit test for the maxq
# mq = max_heapq()
# mq.push('1', 1)
# print(mq.max())
# mq.push('2', 5)
# print(mq.max())
# print(mq.pop())
# mq.push('3', 7)
# mq.push('4', 3)
# mq.push('5', 6)
# mq.push('6', 2)
# print(mq.max())
# mq.remove('3')
# print(mq.max())
# print(mq.pop()) # poped '5'
# print(mq.max())
# mq.remove('4')
# print(mq.pop())
# print(mq.pop())
# print(mq.pop())

class reverse_rw_3:
    # reverse random walk for L=3
    def __init__(self, graph,):
        self.S = {}
        self.P = {}
        self.G = graph
        # for n in graph.nodes:
        #     self.S[n.id] = 0
        #     self.P[n.id] = 0

    def reverse_random_walk_single(self, e, target_id, debug=False): # e for \epsilon, target is a persona vertex, Q is max-heapq
        self.S = {}
        self.P = {}
        maxq = max_heapq()
        self.P[target_id] = 1
        self.S[target_id] = 1
        maxq.push(target_id, 1)
        # (1): we treat all L-nodes' p as 0
        # (2): we treat all 1-unlabeled-nodes' p as 0
        # (3): we do not push any untouched node (with p=0) into the heap
        while maxq.max()[0] >= e/3:
            p_w, w_nid = maxq.pop()
            w = self.G[w_nid]
            if debug: print(w_nid, type(w_nid))
            for u in w.in_nodes:
                u_nid = u.id
                # update S
                delta_s = p_w / u.out_deg()
                if self.S.get(u_nid) is None: self.S[u_nid] = 0
                self.S[u_nid] += delta_s
                # update P (optional)
                if u.layer < 3: # (1)
                    if self.P.get(u_nid) is None: self.P[u_nid] = 0
                    self.P[u_nid] += delta_s
                    maxq.push(u_nid, self.P[u_nid]) # may be update, not real push
            # clean w's p
            self.P[w_nid] = 0
        # pushing end, read S for the approximated results
        return deepcopy(self.S)
    
    def reverse_random_walk_all(self, e, persona_order, unlabeled_uidxs):
        # -> a matrix ordered same as the persona, U*P, simulates the walk_probs
        results_S = []
        for pidx in tqdm(persona_order, total=len(persona_order)):
            pnode_id = 'p' + str(pidx)
            results_S.append(self.reverse_random_walk_single(e, pnode_id))
        # make a matrix with U*P, ordered as (unlabeled_uidxs, persona_order)
        affinities = []
        for uidx in unlabeled_uidxs:
            node_id = 'u'+str(uidx)
            tmp = []
            for i in range(len(persona_order)):
                tmp.append(results_S[i].get(node_id, 1e-12))
            affinities.append(tmp)
        affinities = np.array(affinities)
        # print(affinities.shape)
        assert affinities.shape == (len(unlabeled_uidxs), len(persona_order))
        return affinities       
         
# def rev_rw_unit_test():
    ## unit test
    ## test graph
    # # u0 ---- t0 ---- u_0
    # # u1 ----/  \---- u_1
    # #           \---- l4 ---
    # #            /          \
    # # u2 ---- t1 ---- u_2    --\
    # #           \---- l3 ----- p0
    # #                   \---- p1
    # all_nodes_test = []
    # nodes_dict_test = {}
    # u0 = make_node('u0', NTYP_U, 3, all_nodes_test, nodes_dict_test)
    # u1 = make_node('u1', NTYP_U, 3, all_nodes_test, nodes_dict_test)
    # u2 = make_node('u2', NTYP_U, 3, all_nodes_test, nodes_dict_test)

    # t0 = make_node('t0', NTYP_V, 2, all_nodes_test, nodes_dict_test)
    # t1 = make_node('t1', NTYP_V, 2, all_nodes_test, nodes_dict_test)

    # u_0 = make_node('u_0', NTYP_U, 1, all_nodes_test, nodes_dict_test)
    # u_1 = make_node('u_1', NTYP_U, 1, all_nodes_test, nodes_dict_test)
    # u_2 = make_node('u_2', NTYP_U, 1, all_nodes_test, nodes_dict_test)
    # l3 = make_node('l3', NTYP_L, 1, all_nodes_test, nodes_dict_test)
    # l4 = make_node('l4', NTYP_L, 1, all_nodes_test, nodes_dict_test)

    # p0 = make_node('p0', NTYP_P, 0, all_nodes_test, nodes_dict_test)
    # p1 = make_node('p1', NTYP_P, 0, all_nodes_test, nodes_dict_test)

    # u0.add_out_nodes([t0])
    # u1.add_out_nodes([t0])
    # u2.add_out_nodes([t1])

    # t0.add_in_nodes([u0, u1])
    # t0.add_out_nodes([u_0, u_1, l4])
    # t1.add_in_nodes([u2])
    # t1.add_out_nodes([u_2, l3, l4])

    # l3.add_in_nodes([t1])
    # l3.add_out_nodes([p0, p1])
    # l4.add_in_nodes([t0, t1])
    # l4.add_out_nodes([p0])

    # p0.add_in_nodes([l3, l4])
    # p1.add_in_nodes([l3])

    # test_graph = graph(all_nodes_test)
    # rrw = reverse_rw_3(test_graph)
    # print(rrw.reverse_random_walk_single(0, 'p1', debug=True))

# Iterative_Sampler_rev: the reverse version of the Iterative_Sampler
class Iterative_Sampler_rev:
    def __init__(self, G_user, G_item, oracle, sample_scope, predefined_persona_list, e, col_refine=0.0):
        ## added
        self.e = e # error control
        # self.persona2idx = persona2idx # should align with the graph's order
        ## original
        self.G_user = deepcopy(G_user)
        self.G_uidx2tidx = {k:list(v) for k,v in G_user.items()}
        self.G_item = deepcopy(G_item)
        self.sample_scope = sample_scope # [uidx], sampling scope
        self.oracle = oracle
        self.labeled_GT = {} # subset of the GT, {uidx: [persona]}
        self.predefined_persona_list = predefined_persona_list
        # 1
        self.known_items = set()
        # 3
        # if R is None: self.R = graph2transP(self.G_user, self.G_item)
        # else: self.R = R
        self.col_refine = col_refine
        self.labeled_uidxs = [] # ordered

    def update_labeled_uidxs(self,):
        self.labeled_uidxs = list(self.labeled_GT.keys())
        self.labeled_uidxs.sort()
    
    def update_known_items(self,):
        # should be called after self.update_labeled_uidxs()
        for uidx in self.labeled_uidxs:
            self.known_items.update(self.G_user[uidx])
    
    def trivial_rank(self,):
        # randomly shuffle as a baseline
        unlabeled_uidx = np.setdiff1d(self.sample_scope, list(self.labeled_GT.keys()))
        random.shuffle(unlabeled_uidx)
        return unlabeled_uidx
    
    def criteria_rank_1(self,):
        # rank the unlabeled users according to the new-item coverage
        # -> rank_res: [uidx]; ordered list of unlabeled users, most previous most important
        unlabeled_uidx = np.setdiff1d(self.sample_scope, list(self.labeled_GT.keys()))
        random.shuffle(unlabeled_uidx)
        delta_degrees = {uidx:0 for uidx in unlabeled_uidx} # {uidx: delta_degree}
        for uidx in unlabeled_uidx:
            # for tidx in self.G_user[uidx]:
            #     if tidx in self.known_items: continue
            #     delta_degrees[uidx] += len(G_item[tidx])
            #     # delta_degrees[uidx] += 1
            delta_ = len(set(self.G_user[uidx]).difference(self.known_items))
            delta_degrees[uidx] = delta_

        rank_res = sorted(delta_degrees, key=lambda x:delta_degrees[x], reverse=False)
        return rank_res

    def criteria_rank_3(self,):
        # rank the unlabeled users according to their distribution criteria
        # -> rank_res: [uidx]; ordered list of unlabeled users, most previous most important

        if self.labeled_GT == {}: return self.trivial_rank() # if no seed then randomly sample some as the seed
        
        unlabeled_uidx = np.setdiff1d(self.sample_scope, list(self.labeled_GT.keys()))
        current_dict = {p:0 for p in self.predefined_persona_list}
        
        # make current persona distribution
        for uidx, ps in self.labeled_GT.items():
            for p in ps:
                current_dict[p] += 1
        current_distribution = np.array(list(current_dict.values())) * 1.0 # np.array
        current_distribution /= np.sum(current_distribution) # normalize
        assert np.isclose(np.sum(current_distribution), 1.0) # check

        current_label_matrix, p_list, p2idx = pagerank.assemble_label_matrix(self.labeled_GT, self.labeled_uidxs)
        
        ### replaced
        # # make persona distribution for each user
        # pageranker = PageRank(self.R, unlabeled=unlabeled_uidx)
        # walk_probs = pageranker.run(max_itr=1) # N*U
        # rank_res_labeled = walk_probs[self.labeled_uidxs,:] # L*U
        # rank_res_labeled = rank_res_labeled / np.sum(rank_res_labeled, axis=0, keepdims=True) # normalize each column, L*U
        # assert np.all(np.isclose(np.sum(rank_res_labeled, axis=0), 1.0)) # check
        # rank_res_labeled = rank_res_labeled.T # U*L
        # scores = rank_res_labeled @ current_label_matrix # (U*L)*(L*C)=U*C
        ### replaced to:
        G_uidx2pidx = {uidx:[p2idx[p] for p in ps] for uidx,ps in self.labeled_GT.items()}
        the_graph = construct_graph(unlabeled_uidx, self.labeled_uidxs, G_uidx2pidx, self.G_uidx2tidx)
        rrw = reverse_rw_3(the_graph)
        scores = rrw.reverse_random_walk_all(self.e, range(len(p_list)), unlabeled_uidx) # U*C, ordered: (unlabeled_uidx, p2idx)
        ### replaced end
        # refine the label matrix-2 (optional)
        # making label_matrix_tuned
        label_matrix_tuned = deepcopy(current_label_matrix)
        
        col_refine = self.col_refine
        if col_refine > 0:
            col_sums = np.sum(label_matrix_tuned, axis=0, keepdims=True)
            tuning_factor = 1/col_sums
            tuning_factor = tuning_factor / np.min(tuning_factor)
            tuning_factor = tuning_factor ** col_refine
            scores = scores * tuning_factor

        # make whole scores for all personas (span)
        whole_scores = np.zeros([len(unlabeled_uidx), len(self.predefined_persona_list)]) # U*C'
        for i,p in enumerate(self.predefined_persona_list):
            # the order of p_list align with p2idx's keys
            if p not in p_list: continue # skip, keep all values 0.0
            whole_scores[:,i] = scores[:, p2idx[p]]
        
        whole_scores /= np.sum(whole_scores, axis=1, keepdims=True) # normalize each row, U*C'

        # calculate the KL divergence
        kld = []
        for i in range(len(unlabeled_uidx)):
            tmp = scipy.stats.entropy(whole_scores[i, :]) + scipy.stats.entropy(current_distribution, whole_scores[i, :] + 1e-16) # important
            # tmp = scipy.stats.entropy(whole_scores[i, :]) + scipy.stats.entropy(whole_scores[i, :] + 1e-16, current_distribution) # important
            # tmp = scipy.stats.entropy(current_distribution, whole_scores[i, :])
            # tmp = scipy.stats.entropy(whole_scores[i, :])
            kld.append(tmp)
        sorted_ii = np.argsort(kld)[::-1] # resort as reversed, a middle var
        rank_res = unlabeled_uidx[sorted_ii]

        return rank_res
    
    def run(self, sample_amount, chunk_size, seed_amount=0, criteria=3):

        # seeding sample
        if seed_amount > 0:
            random_rank = self.trivial_rank()
            seed_GT = self.oracle.query(random_rank[:seed_amount])
            self.labeled_GT.update(seed_GT)
            sample_amount -= seed_amount
        
        for i in range(sample_amount // chunk_size):
            # update the decision variables first before decision
            self.update_labeled_uidxs() # 3: update labeled uidx
            self.update_known_items()
            # new rank
            if criteria == 3:
                rank_res = self.criteria_rank_3()
            elif criteria == 1:
                rank_res = self.criteria_rank_1()
            else:
                assert False, f'not supported criteria={criteria}'

            sample_batch = rank_res[:chunk_size] # [uidx]
            sample_GT = self.oracle.query(sample_batch)
            self.labeled_GT.update(sample_GT)

        return self.labeled_GT

def persona_prediction_rev(
    e, # error rate
    G_user,
    sampled_GT, # {uidx:[persona,]}; sampled ground truth
    user_num,   # int; len(G_user)
    # hyper parameters
    # row_refine = True,
    col_refine = 0.5,
    # knn_k = 0,
    timer=False,
):

    # making label_matrix_tuned
    labeled_uidxs = list(sampled_GT.keys())
    labeled_uidxs.sort()

    unlabeled_uidxs = np.setdiff1d(np.arange(user_num), labeled_uidxs) # unlabeled user index set, ordered by uidx
    labeled_user_personas = sampled_GT

    label_matrix, persona_list, persona2idx = pagerank.assemble_label_matrix(labeled_user_personas, labeled_uidxs)
    label_matrix_tuned = deepcopy(label_matrix)

    # construct the graph
    G_uidx2tidx = {k:list(v) for k,v in G_user.items()}
    G_uidx2pidx = {uidx:[persona2idx[p] for p in ps] for uidx,ps in sampled_GT.items()}
    the_graph = construct_graph(unlabeled_uidxs, labeled_uidxs, G_uidx2pidx, G_uidx2tidx)
    start_time = time.time()
    rrw = reverse_rw_3(the_graph)
    scores = rrw.reverse_random_walk_all(e, range(len(persona_list)), unlabeled_uidxs)
    end_time = time.time()
    cost_time = end_time - start_time

    # refine the label matrix-2 (optional)
    if col_refine > 0:
        col_sums = np.sum(label_matrix_tuned, axis=0, keepdims=True)
        tuning_factor = 1/col_sums
        tuning_factor = tuning_factor / np.min(tuning_factor)
        tuning_factor = tuning_factor ** col_refine
        scores = scores * tuning_factor
    
    # fined results
    persona_probs = scores
    if timer:
        return persona_probs, label_matrix, persona_list, persona2idx, cost_time
    else:
        return persona_probs, label_matrix, persona_list, persona2idx,
