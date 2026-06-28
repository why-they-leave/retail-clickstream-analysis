import random
from copy import deepcopy

import numpy as np
import scipy
from tqdm import tqdm


def sparse_multiply(A, B):
    # A @ B for sparse matrix A and B
    # A_ = scipy.sparse.csc_matrix(A)
    # B_ = scipy.sparse.csc_matrix(B)
    A_ = A.tocsc()
    B_ = B.tocsc()
    return A_ @ B_

# graph2transP: convert a graph to random walk transformation probability
def graph2transP(G_user_denoise, G_item_denoise, debug=False, sparse=False):
    # give some all checkings for performance

    user_num = len(G_user_denoise)
    item_num = len(G_item_denoise)

    # make U2I
    if not sparse:
        U2I = np.zeros([user_num, item_num])
    else:
        U2I = scipy.sparse.lil_matrix((user_num, item_num))
    if debug: print('Constructing U2I', end='')
    for uidx in range(user_num):
        bought_items = np.array(list(G_user_denoise[uidx]))
        if len(bought_items) == 0: # if all bought items removed when denoising, then assumes buys all items (except the deleted ones), for the computing correctness
            bought_items = list(G_item_denoise.keys())
            bought_items.sort()
        trans_p = 1/len(bought_items)
        U2I[uidx,bought_items] = trans_p
        if debug and uidx%(user_num//10) == 0: print('.', end='')
    # assert np.all(np.isclose(np.sum(U2I, axis=1), 1.0)) # check

    # make I2U
    if not sparse:
        I2U = np.zeros([item_num, user_num])
    else:
        I2U = scipy.sparse.lil_matrix((item_num, user_num))
    if debug: print('\nConstructing I2U', end='')
    for tidx in range(item_num):
        bought_users = list(G_item_denoise[tidx])
        # if bought_users an non-empty set, trans p > 0.0
        if bought_users:
            trans_p = 1/len(bought_users)
            I2U[tidx,bought_users] = trans_p
        # else, average probability
        else:
            bought_users = np.array(bought_users)
            I2U[tidx,:] = 1/user_num
        if debug and tidx%(item_num//10) == 0: print('.', end='')
    # assert np.all(np.isclose(np.sum(I2U, axis=1), 1.0)) # check

    # make P
    if not sparse:
        P = U2I @ I2U
    else:
        P = sparse_multiply(U2I, I2U).toarray()
    if debug: assert np.all(np.isclose(np.sum(P, axis=1), 1.0)) # check
    if debug: print('\nGet P')

    # make R
    beta = 0.999 # hyper parameter
    v = np.ones([user_num,1])
    e = 1/user_num * v
    R = beta*P + (1-beta)*v@e.T
    if debug: assert np.all(np.isclose(np.sum(R, axis=1), 1.0)) # check
    if debug: assert R.shape == (user_num, user_num)
    if debug: print('Get R, done.')

    return R


# PageRank: implements the random walk based multi-label prediction algorithm
class PageRank:
    def __init__(self, R, unlabeled):
        # R: np.array; transformation probability matrix, N*N
        # unlabeled: [uidx,]; the list of unlabeled user idxs
        self.R = deepcopy(R)
        self.N = R.shape[0]
        self.U = len(unlabeled) # number of unlabeled users

        self.pi = np.zeros([self.N, self.U]) # N*U, starting probabilities for each unlabeled user
        self.unlabeled = np.array(unlabeled)
        self.pi[self.unlabeled, np.arange(self.U)] = 1 # pi is sparse

    def run(self, conv_th=1e-10, max_itr=100):
        # -> new_pi: np.array, N*U; resulted probability for each unlabeled users
        last_pi = deepcopy(self.pi)
        for itr in tqdm(range(max_itr)):
            new_pi = self.R.T @ last_pi # we can sparsify the self.R.T
            # check convergence
            if np.linalg.norm(new_pi - last_pi) <= conv_th:
                break
            last_pi = new_pi
        return new_pi


# Persona_Oracle: act as an external oracle
class Persona_Oracle:
    # a tricky implementation of the oracle
    # user_ids: [uid]; the original user ids, uid = user_ids[uidx]
    def __init__(self, kvpairs, user_ids):
        # kvpairs: {uid: [persona]}
        self.kvpairs = kvpairs
        self.user_ids = user_ids

    def query(self, user_idxs,):
        # query the persona for sampled user ids
        # user_idxs: numpy array, ordered; the user idxs to be sampled
        # assumption: no unrepresentable user is queried
        # -> response: {uidx: [persona]}
        response = {}
        for uidx in user_idxs:
            uid = self.user_ids[uidx]
            p_s = self.kvpairs[uid]

            assert 'Unrepresentable' not in p_s, p_s # check

            response[uidx] = p_s

        return response


# assemble_label_matrix: according to supervision, construct the label matrix with shape L*C
def assemble_label_matrix(labeled_user_personas, labeled_uidxs):
    # order the current personas
    persona2idx = {}
    persona_list = []
    for uidx, ps in labeled_user_personas.items():
        for p in ps:
            if persona2idx.get(p) is None:
                persona2idx[p] = len(persona_list)
                persona_list.append(p)
    assert len(persona_list)==len(persona2idx) # check

    # assemble the label_matrix
    label_matrix = np.zeros([len(labeled_uidxs), len(persona_list)]) # L * C
    check_summer = 0
    for i in range(len(labeled_uidxs)):
        uidx = labeled_uidxs[i]
        ps = labeled_user_personas[uidx]
        ps = list(set(ps)) # remove redundant if any
        pidxs = [persona2idx[p] for p in ps]
        label_matrix[i,pidxs] = 1
        # check
        check_summer += len(ps)
    assert np.sum(label_matrix) == check_summer, f'{np.sum(label_matrix)} != {check_summer}' # check

    return label_matrix, persona_list, persona2idx


# Iterative_Sampler: class provides different supervision sampling methods
class Iterative_Sampler:
    def __init__(self, G_user, G_item, oracle, sample_scope, predefined_persona_list, R=None):
        self.G_user = deepcopy(G_user)
        self.G_item = deepcopy(G_item)
        self.sample_scope = sample_scope # [uidx], sampling scope
        self.oracle = oracle
        self.labeled_GT = {} # subset of the GT, {uidx: [persona]}
        self.predefined_persona_list = predefined_persona_list
        # 1
        self.known_items = set() # {tidx}; set of known items
        # 3
        if R is None: self.R = graph2transP(self.G_user, self.G_item)
        else: self.R = R
        self.labeled_uidxs = [] # ordered

    def update_labeled_uidxs(self,):
        self.labeled_uidxs = list(self.labeled_GT.keys())
        self.labeled_uidxs.sort()

    def update_known_items(self,):
        # should be called after self.update_labeled_uidxs()
        for uidx in self.labeled_uidxs:
            self.known_items.update(self.G_user[uidx])

    def remove_overly_focused(self, remove_amount):
        # call first, before any iterations
        # remove all overly focused users from the sampling scope
        unlabeled_uidx = self.sample_scope
        R = graph2transP(self.G_user, self.G_item)
        pageranker = PageRank(R, unlabeled=self.sample_scope)
        pi = pageranker.run(max_itr=1) # pi is N*U
        being_focused = {uidx: np.sum(pi[uidx, :]) for uidx in unlabeled_uidx} # the smaller the better
        rank = sorted(being_focused, key=lambda x:being_focused[x],)
        self.sample_scope = deepcopy(rank[:-remove_amount]) # avoiding the overly focused user

    def remove_too_common_users(self, remove_amount):
        # call first, before any iterations
        unlabeled_uidx = self.sample_scope
        being_focused = {uidx: len(G_user[uidx]) for uidx in unlabeled_uidx if len(G_user[uidx]) >= 0} # the smaller the better
        rank = sorted(being_focused, key=lambda x:being_focused[x],)
        self.sample_scope = deepcopy(rank[:-remove_amount]) # avoiding the too common user

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
            delta_degrees[uidx] += delta_

        rank_res = sorted(delta_degrees, key=lambda x:delta_degrees[x], reverse=False)
        return rank_res

    # def criteria_rank_2(self,):
    #     # abandon
    #     pass

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

        current_label_matrix, p_list, p2idx = assemble_label_matrix(self.labeled_GT, self.labeled_uidxs)
        # test
        # print(current_label_matrix.shape)
        # current_label_matrix = current_label_matrix / np.sum(current_label_matrix, axis=1, keepdims=True)

        # make persona distribution for each user
        pageranker = PageRank(self.R, unlabeled=unlabeled_uidx)
        walk_probs = pageranker.run(max_itr=1) # N*U
        rank_res_labeled = walk_probs[self.labeled_uidxs,:] # L*U
        rank_res_labeled = rank_res_labeled / np.sum(rank_res_labeled, axis=0, keepdims=True) # normalize each column, L*NL
        assert np.all(np.isclose(np.sum(rank_res_labeled, axis=0), 1.0)) # check
        rank_res_labeled = rank_res_labeled.T # U*L
        scores = rank_res_labeled @ current_label_matrix # (U*L)*(L*C)=U*C

        # make whole scores for all personas
        whole_scores = np.zeros([len(unlabeled_uidx), len(self.predefined_persona_list)]) # U*C'
        for i,p in enumerate(self.predefined_persona_list):
            # the order of p_list align with p2idx's keys
            if p not in p_list: continue # skip, keep all values 0.0
            whole_scores[:,i] = scores[:, p2idx[p]]

        whole_scores /= np.sum(whole_scores, axis=1, keepdims=True) # normalize each row, U*C'

        # calculate the KL divergence
        kld = []
        for i in range(len(unlabeled_uidx)):
            tmp = scipy.stats.entropy(whole_scores[i, :]) + scipy.stats.entropy(current_distribution, whole_scores[i, :]) # important
            # tmp = scipy.stats.entropy(current_distribution, whole_scores[i, :])
            # tmp = scipy.stats.entropy(whole_scores[i, :])
            kld.append(tmp)
        sorted_ii = np.argsort(kld)[::-1] # resort as reversed, a middle var
        rank_res = unlabeled_uidx[sorted_ii]

        return rank_res

    # def aggregated_rank(self, rank_method=3):
    #     # aggregate the ranks
    #     # abandon
    #     pass

    def run(self, sample_amount, chunk_size, remove_overf=0, seed_amount=0, criteria=3):
        # remove overly focused
        if remove_overf > 0:
            self.remove_overly_focused(remove_overf)

        # seeding sample
        if seed_amount > 0:
            random_rank = self.trivial_rank()
            seed_GT = self.oracle.query(random_rank[:seed_amount])
            self.labeled_GT.update(seed_GT)
            sample_amount -= seed_amount

        for i in range(sample_amount // chunk_size):
            # update the decision variables first before decision
            self.update_labeled_uidxs() # 3: update labeled uidx
            self.update_known_items() # 1: update known items

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

def persona_prediction(
    R,  # transformation matrix
    sampled_GT, # {uidx:[persona,]}; sampled ground truth
    user_num,   # int; len(G_user)
    # hyper parameters
    row_refine = True,
    col_refine = 0.5,
    knn_k = 0,
):
    # making label_matrix_tuned
    labeled_uidxs = list(sampled_GT.keys())
    labeled_uidxs.sort()

    unlabeled_uidxs = np.setdiff1d(np.arange(user_num), labeled_uidxs) # unlabeled user index set, ordered by uidx
    labeled_user_personas = sampled_GT

    label_matrix, persona_list, persona2idx = assemble_label_matrix(labeled_user_personas, labeled_uidxs)

    label_matrix_tuned = deepcopy(label_matrix)

    # refine the label matrix-1 (optional)
    if row_refine:
        row_sums = np.sum(label_matrix_tuned, axis=1, keepdims=True)
        label_matrix_tuned = label_matrix_tuned / row_sums

    # refine the label matrix-2 (optional)
    if col_refine > 0:
        col_sums = np.sum(label_matrix_tuned, axis=0, keepdims=True)
        tuning_factor = 1/col_sums
        tuning_factor = tuning_factor / np.min(tuning_factor)
        tuning_factor = tuning_factor ** col_refine
        label_matrix_tuned = label_matrix_tuned * tuning_factor

    # Page Rank execution
    pageranker = PageRank(R, unlabeled=unlabeled_uidxs)
    rank_res = pageranker.run(max_itr=1)

    rank_res_labeled = rank_res[labeled_uidxs,:] # L*NL
    rank_res_labeled = rank_res_labeled / np.sum(rank_res_labeled, axis=0, keepdims=True) # normalize each column, L*NL
    assert np.all(np.isclose(np.sum(rank_res_labeled, axis=0), 1.0)) # check
    rank_res_labeled = rank_res_labeled.T # NL*L

    # apply KNN (optional)
    if knn_k > 0:
        k=knn_k
        knn_indices = np.argsort(rank_res_labeled, axis=1)[:, -k:] # NL*K, select nearest neighbors with highest trans probability
        row_indices = np.arange(rank_res_labeled.shape[0])[:, None] # NL*1, indices
        knn_mask = np.zeros_like(rank_res_labeled, dtype=bool)
        knn_mask[row_indices, knn_indices] = True # set the k-nearest neighbors' filter
        rank_res_labeled = rank_res_labeled * knn_mask # removes all other probabilities except the K-nearest neighbors
        rank_res_labeled = rank_res_labeled / np.sum(rank_res_labeled, axis=1, keepdims=True) # normalize each row (for each non-labeled sample)
        assert np.all(np.isclose(np.sum(rank_res_labeled, axis=1), 1.0)) # check

    # fined results
    persona_probs = rank_res_labeled @ label_matrix_tuned
    return persona_probs, label_matrix, persona_list, persona2idx
