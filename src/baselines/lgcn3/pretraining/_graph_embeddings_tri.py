import json
import scipy as sp
import scipy.sparse.linalg
from numpy import *

DATASET = 2             # 0 for Amazon, 1 for Movielens, 2 for MBA, 3 for Instacart, 4 for Instacart_full
FREQUENCY = 128         # dimensionality of the base, the 'cutoff' frequency, relates to the de-noising level, should be tuned
FREQUENCY_U = [100, 300, 100, 100, 100][DATASET]   # dimensionality of the base of the user graph (no use for 1-d)
FREQUENCY_I = [50, 200, 50, 50, 50][DATASET]    # dimensionality of the base of the user graph (no use for 1-d)
GRAPH_CONV = ['1d', '2d'][0]            # 0 for 1d convolution and 1 for 2d
APPROXIMATE = False # False
Dataset = ['Amazon', 'Movielens', 'MBA', 'Instacart', 'Instacart_full'][DATASET]
tolerant = 0.1 ** 5
epsilon = 0.1 ** 10

root = '../../../data'
u2t_train_path = root + '/tri_graph_uidx2tidx_train.json'
t2p_path = root + '/tri_graph_tidx2pidx.json'

if APPROXIMATE:
    # approaximated case
    u2p_path = root + '/gplr_res/tri_graph_uidx2pidx_app_e_0.1.json'
    path_save = root + '/graph_embeddings_' + GRAPH_CONV + '_tri_approach.json'
else:
    # normal case
    u2p_path = root + '/tri_graph_uidx2pidx.json'
    path_save = root + '/graph_embeddings_' + GRAPH_CONV + '_tri.json'

print('Reading data...')
with open(u2p_path, 'r') as f:
    tri_graph_uidx2pidx = json.load(f)
with open(t2p_path, 'r') as f:
    tri_graph_tidx2pidx = json.load(f)
with open(u2t_train_path) as f:
    tri_graph_uidx2tidx_train = json.load(f)

# trans key to int
tri_graph_uidx2pidx = {int(k):v for k,v in tri_graph_uidx2pidx.items()}
tri_graph_tidx2pidx = {int(k):v for k,v in tri_graph_tidx2pidx.items()}
tri_graph_uidx2tidx_train = {int(k):v for k,v in tri_graph_uidx2tidx_train.items()}

user_number = len(tri_graph_uidx2tidx_train)
item_number = len(tri_graph_tidx2pidx)
assert item_number == max(list(tri_graph_tidx2pidx.keys())) + 1

if DATASET == 2:
    persona_number = 20
elif DATASET == 3 or DATASET == 4:
    persona_number = 51 # 20
print(f'persona_number:{persona_number}')

if GRAPH_CONV == '1d':
    # todo: need change a lot
    print('Initializing...')
    matrix_l = user_number + item_number + persona_number
    A = sp.sparse.lil_matrix((matrix_l, matrix_l)) #M+N+P * M+N+P
    D = sp.sparse.lil_matrix((matrix_l, matrix_l)) #M+N+P * M+N+P, should be diagnal
    I = sp.sparse.lil_matrix((matrix_l, matrix_l)) #M+N+P * M+N+P
    for i in range(matrix_l): I[i, i] = 1

    # constructing the laplacian matrices
    print('Constructing the laplacian matrices...')
    for uidx, ps in tri_graph_uidx2pidx.items():
        for pidx in ps:
            A[uidx, user_number+item_number+pidx] = 1
            A[user_number+item_number+pidx, uidx] = 1
            D[uidx, uidx] += 1
            D[user_number+item_number+pidx, user_number+item_number+pidx] += 1
    for uidx, ts in tri_graph_uidx2tidx_train.items():
        for tidx in ts:
            A[uidx, user_number+tidx] = 1
            A[user_number+tidx, uidx] = 1
            D[uidx, uidx] += 1
            D[user_number+tidx, user_number+tidx] += 1
    for tidx, ps in tri_graph_tidx2pidx.items():
        for pidx in ps: # skip if empty
            A[user_number+tidx, user_number+item_number+pidx] = 1
            A[user_number+item_number+pidx, user_number+tidx] = 1
            D[user_number+tidx, user_number+tidx] += 1
            D[user_number+item_number+pidx, user_number+item_number+pidx] += 1

    for l in range(user_number + item_number + persona_number):
        D[l, l] = 1.0 / max(sqrt(D[l, l]), epsilon)
    L = I - D * A * D

    #eigenvalue factorization
    print('Decomposing the laplacian matrices...')
    [Lamda, graph_embeddings] = sp.sparse.linalg.eigsh(L, k = FREQUENCY, which='SM', tol = tolerant) # returns the first 128 most dominant eigen vectors
    print(Lamda[0:10])

    # print(f'debug, persona_number:{persona_number}')
    # quit()

    print('Saving features...')
    f = open(path_save, 'w')
    jsObj = json.dumps(graph_embeddings.tolist())
    f.write(jsObj)
    f.close()
else:
    print('Not supported for non 1-d')
