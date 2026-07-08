"""표준 LightGCN을 tri-graph(유저-상품-세그먼트)에 적용한 모델 (Issue #30).

model_LGCN_tri.py는 사전학습된 frequency embedding을 쓰는 spectral 방식이라
우리 데이터로는 바로 못 쓴다. 이 모델은 대신 dense2sparse.py의
propagation_matrix_tri()가 만든 정규화 인접행렬을 레이어마다 그대로 곱하는
표준 LightGCN 전파를 쓴다 — 학습되는 커널/변환 행렬이 없는 게 핵심 단순화다.
설계 배경은 docs/LIGHTGCN_TRI_MODEL_DESIGN.md 참고.
"""

import tensorflow as tf

# 모듈 임포트 시점에 즉시 비활성화한다 — __init__ 안에서 호출하면, 호출자가
# __init__보다 먼저 만든 텐서(예: sparse_graph)가 이미 EagerTensor로 굳어버려
# 이후 그래프 모드 연산에 못 들어가는 문제가 생긴다 (테스트로 확인됨).
tf.compat.v1.disable_eager_execution()

_OPTIMIZERS = {
    "SGD": tf.compat.v1.train.GradientDescentOptimizer,
    "RMSProp": tf.compat.v1.train.RMSPropOptimizer,
    "Adam": tf.compat.v1.train.AdamOptimizer,
    "Adagrad": tf.compat.v1.train.AdagradOptimizer,
}


_LAYER_WEIGHT_SCHEMES = {
    # 표준 LightGCN: 모든 레이어(0~layer)에 동일한 가중치 1/(layer+1)
    "uniform": lambda layer: [1 / (layer + 1)] * (layer + 1),
    # 레이어 인덱스가 커질수록 가중치가 줄어듦(1/(l+1)) — 원본 임베딩(레이어0)에
    # 편중된 방식. 원래 uniform 대신 실수로 쓰였던 공식이지만, 이 프로젝트
    # 데이터에서 uniform보다 성능이 나아 정식 비교 대상으로 남겨둔다 (#37).
    "decay": lambda layer: [1 / (l + 1) for l in range(layer + 1)],
}


class model_LightGCN_tri(object):
    def __init__(
        self,
        n_users,
        n_items,
        n_personas,
        lr,
        lamda,
        emb_dim,
        layer,
        sparse_graph,
        optimization,
        layer_weight_scheme="uniform",
    ):
        if optimization not in _OPTIMIZERS:
            raise ValueError(f"알 수 없는 optimization: {optimization}")
        if layer_weight_scheme not in _LAYER_WEIGHT_SCHEMES:
            raise ValueError(f"알 수 없는 layer_weight_scheme: {layer_weight_scheme}")

        self.model_name = "LightGCN_tri"
        self.n_users = n_users
        self.n_items = n_items
        self.n_personas = n_personas
        self.lr = lr
        self.lamda = lamda
        self.emb_dim = emb_dim
        self.layer = layer
        self.sparse_graph = sparse_graph

        # 학습용 배치 입력 (BPR: 유저별로 실제 산 상품 하나 + 안 산 상품 하나를 짝지어 넣음)
        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        # 추론(top-k 추천)에서만 씀 — 학습에 이미 나온 상품은 큰 음수를 더해 순위 맨 뒤로 보내는 용도.
        # 실제 마스킹 값은 호출하는 쪽(run_lightgcn.py)이 채워 넣는다, 이 클래스는 자리만 정의.
        self.items_in_train_data = tf.compat.v1.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.compat.v1.placeholder(tf.int32, shape=())
        # 이 모델은 dropout을 안 쓰지만, train_model.py/test_model.py가 모델 종류와 무관하게
        # feed_dict에 model.keep_prob를 항상 넣기 때문에 placeholder 자체는 있어야 한다
        # (값은 실제로 아무 데도 안 쓰임 — 순수 호환용).
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, shape=(None))

        # 재현성(CLAUDE.md 규칙: random_state=42) — seed 없으면 실행마다 초기 임베딩이
        # 달라져 학습 결과를 비교할 수 없다 (CodeRabbit 지적).
        tf.compat.v1.set_random_seed(42)
        self.user_embeddings = tf.Variable(
            tf.random.normal([n_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name="user_embeddings",
        )
        self.item_embeddings = tf.Variable(
            tf.random.normal([n_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name="item_embeddings",
        )
        self.persona_embeddings = tf.Variable(
            tf.random.normal([n_personas, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
            name="persona_embeddings",
        )

        # 레이어 결합 가중치 — 기본은 표준 LightGCN의 균등 평균(uniform), 학습되는
        # 가중치 없음. layer_weight_scheme으로 선택 가능 (#37, _LAYER_WEIGHT_SCHEMES 참고).
        self.layer_weight = _LAYER_WEIGHT_SCHEMES[layer_weight_scheme](layer)
        layer_weight = self.layer_weight
        # 유저/상품/세그먼트 임베딩을 하나의 큰 행렬로 합쳐서 그래프 전파에 쓴다
        # (sparse_graph의 행/열 순서가 [유저, 상품, 세그먼트] 순으로 만들어져 있음, dense2sparse.py 참고)
        embeddings = tf.concat(
            [self.user_embeddings, self.item_embeddings, self.persona_embeddings], axis=0
        )
        all_embeddings = embeddings * layer_weight[0]  # 0번째 레이어(자기 자신)도 평균에 포함
        for l in range(layer):
            # 표준 LightGCN 전파 그 자체: E^(l+1) = 정규화_인접행렬 @ E^(l)
            # model_LGCN_tri.py처럼 레이어마다 학습되는 kernel/변환행렬을 곱하지 않는다 —
            # 이 "곱할 게 없다"는 단순함이 spectral 방식(LGCN_tri) 대비 LightGCN의 핵심 차이다
            embeddings = tf.sparse.sparse_dense_matmul(self.sparse_graph, embeddings)
            all_embeddings += embeddings * layer_weight[l + 1]

        # 전파가 끝난 뒤 다시 유저/상품/세그먼트 파트로 분리 (합쳤던 순서 그대로 되돌림)
        self.user_all_embeddings, self.item_all_embeddings, self.persona_all_embeddings = tf.split(
            all_embeddings, [n_users, n_items, n_personas], axis=0
        )

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        # pos_scores/neg_scores: BPR 학습용 (내적 점수, "산 상품 점수가 안 산 상품보다 높아야 함")
        self.pos_scores = tf.reduce_sum(tf.multiply(self.u_embeddings, self.pos_i_embeddings), 1)
        self.neg_scores = tf.reduce_sum(tf.multiply(self.u_embeddings, self.neg_i_embeddings), 1)
        # all_ratings/top_items: 학습이 아니라 평가·추천용 — 유저 하나당 전체 상품 점수를 매겨
        # 상위 top_k개를 뽑는다. items_in_train_data로 이미 학습에서 본 상품을 순위 밖으로 밀어낸다.
        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_b=True)
        self.all_ratings += self.items_in_train_data
        _top_k = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True)
        self.top_items = _top_k.indices
        self.top_scores = _top_k.values  # run_lightgcn.py의 CSV 저장(score 컬럼)에 필요

        self.loss = self.bpr_loss(self.pos_scores, self.neg_scores)
        # 세그먼트 임베딩도 정규화 대상에 포함 — 안 그러면 세그먼트 쪽만 무한정 커질 수 있음
        self.loss += self.lamda * self.regularization(
            [
                self.u_embeddings,
                self.pos_i_embeddings,
                self.neg_i_embeddings,
                self.persona_embeddings,
            ]
        )

        self.opt = _OPTIMIZERS[optimization](learning_rate=lr)
        # 학습 대상은 임베딩 3개뿐 — model_LGCN_tri.py와 달리 kernel/transformation/pooling
        # 가중치가 아예 없어서(표준 LightGCN) var_list가 훨씬 짧다
        self.var_list = [self.user_embeddings, self.item_embeddings, self.persona_embeddings]
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

    def bpr_loss(self, pos_scores, neg_scores):
        # log(sigmoid(x)) 대신 수치안정적인 log_sigmoid(x)를 쓴다 — sigmoid가 0으로
        # 포화되면 log(0)=-inf가 될 수 있음 (CodeRabbit 지적)
        maxi = tf.math.log_sigmoid(pos_scores - neg_scores)
        return tf.negative(tf.reduce_sum(maxi))

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list:
            reg += tf.nn.l2_loss(para)
        return reg
