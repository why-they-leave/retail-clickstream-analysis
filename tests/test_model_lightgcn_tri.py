"""model_LightGCN_tri 단위 테스트 (Issue #30, docs/LIGHTGCN_TRI_MODEL_DESIGN.md 참고).

표준 LightGCN 전파(정규화 인접행렬을 그대로 곱하는 방식)로 그래프를 만들고
세션에서 정상 실행되는지 확인한다. 실제 학습 성능이 아니라 "그래프가 에러
없이 빌드/실행되는지"를 확인하는 스모크 테스트 수준이다.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.baselines.lgcn3.model_LightGCN_tri import model_LightGCN_tri

# tf.compat.v1.disable_eager_execution()이 프로세스 전역 상태를 바꾸므로,
# 매 테스트마다 그래프를 리셋해 변수 이름 충돌/그래프 누적을 막는다.


@pytest.fixture(autouse=True)
def reset_graph():
    tf.compat.v1.reset_default_graph()
    yield
    tf.compat.v1.reset_default_graph()


def make_identity_sparse_graph(total_nodes: int) -> tf.SparseTensor:
    """전파해도 임베딩이 그대로 나오는 항등행렬 — 순수 배관(plumbing) 검증용."""
    indices = [[i, i] for i in range(total_nodes)]
    values = [1.0] * total_nodes
    return tf.SparseTensor(indices=indices, values=values, dense_shape=[total_nodes, total_nodes])


class TestModelLightGCNTri:
    def test_builds_and_runs_forward_pass(self):
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=2,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            top_items, top_scores = sess.run(
                [model.top_items, model.top_scores],
                feed_dict={
                    model.users: [0, 1],
                    model.items_in_train_data: np.zeros((2, n_items), dtype=np.float32),
                    model.top_k: 2,
                },
            )
            assert top_items.shape == (2, 2)
            # top_k(sorted=True)라 각 유저 행 내에서 점수가 내림차순이어야 함 —
            # run_lightgcn.py가 이 순서를 그대로 rank로 쓰기 때문에 중요한 속성
            assert (top_scores[:, 0] >= top_scores[:, 1]).all()

    def test_zero_personas_supported(self):
        """bipartite 모드(#35)는 n_personas=0에 가까운 상태로 넘어올 수 있다."""
        n_users, n_items, n_personas = 3, 3, 0
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=1,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            loss = sess.run(
                model.loss,
                feed_dict={
                    model.users: [0],
                    model.pos_items: [0],
                    model.neg_items: [1],
                },
            )
            assert np.isfinite(loss)

    def test_bpr_loss_is_finite_scalar(self):
        n_users, n_items, n_personas = 4, 4, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=1,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            loss = sess.run(
                model.loss,
                feed_dict={model.users: [0, 1], model.pos_items: [0, 1], model.neg_items: [2, 3]},
            )
            assert np.ndim(loss) == 0
            assert np.isfinite(loss)

    def test_one_training_step_reduces_loss_or_stays_finite(self):
        """updates op을 한 번 돌려도 그래프가 안 깨지는지 (실제 수렴 여부는 확인 안 함).

        train_model.py/test_model.py는 모델 종류와 무관하게 feed_dict에
        model.keep_prob를 항상 넣는다 — 이 모델이 dropout을 안 쓰더라도
        placeholder 자체는 있어야 그 공용 코드가 안 깨진다.
        """
        n_users, n_items, n_personas = 4, 4, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=1,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            feed = {
                model.users: [0, 1],
                model.pos_items: [0, 1],
                model.neg_items: [2, 3],
                model.keep_prob: 1,
            }
            _, loss = sess.run([model.updates, model.loss], feed_dict=feed)
            assert np.isfinite(loss)

    def test_bpr_loss_stays_finite_for_extreme_score_gap(self):
        """pos-neg 점수 차가 아주 클 때 sigmoid가 0/1로 포화되면
        log(sigmoid(x))는 -inf가 될 수 있다 — log_sigmoid로 수치안정성 확보.
        """
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=1,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )
        # pos < neg로 점수를 완전히 뒤집어서 sigmoid가 0으로 포화되게 만든다
        # (log(0) = -inf가 되는 조건 — sigmoid가 1로 포화되는 반대 방향은 log(1)=0이라 안전함)
        pos = tf.constant([-100.0], dtype=tf.float32)
        neg = tf.constant([100.0], dtype=tf.float32)
        loss_tensor = model.bpr_loss(pos, neg)

        with tf.compat.v1.Session() as sess:
            loss = sess.run(loss_tensor)
            assert np.isfinite(loss)

    def test_layer_weight_is_uniform_average(self):
        """표준 LightGCN의 레이어 결합은 모든 레이어에 동일한 가중치 1/(K+1)을 쓴다
        (K=layer 수). 레이어 인덱스에 따라 가중치가 줄어드는 방식(1/(l+1))이 아니다.
        """
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        layer = 2
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=layer,
            sparse_graph=sparse_graph,
            optimization="Adam",
        )

        assert len(model.layer_weight) == layer + 1
        expected = 1 / (layer + 1)
        assert all(w == pytest.approx(expected) for w in model.layer_weight)

    def test_embedding_init_is_reproducible_across_runs(self):
        """CLAUDE.md 재현성 규칙(random_state=42) — 임베딩 초기값이 실행마다
        달라지면 안 된다(CodeRabbit 지적: tf.random.normal에 seed 없었음).
        """
        n_users, n_items, n_personas = 3, 3, 2

        def build_and_get_initial_user_embeddings():
            tf.compat.v1.reset_default_graph()
            sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
            model = model_LightGCN_tri(
                n_users=n_users,
                n_items=n_items,
                n_personas=n_personas,
                lr=0.01,
                lamda=0.01,
                emb_dim=4,
                layer=1,
                sparse_graph=sparse_graph,
                optimization="Adam",
            )
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                return sess.run(model.user_embeddings)

        first_run = build_and_get_initial_user_embeddings()
        second_run = build_and_get_initial_user_embeddings()
        assert np.allclose(first_run, second_run)

    def test_layer_weight_scheme_decay_matches_1_over_l_plus_1(self):
        """layer_weight_scheme='decay'는 레이어 인덱스가 커질수록 가중치가
        줄어드는 1/(l+1) 방식 — 원본 임베딩(레이어0)에 편중된 예전 동작을
        의도적 옵션으로 남겨 uniform과 정식으로 비교할 수 있게 한다 (#37).
        """
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        layer = 2
        model = model_LightGCN_tri(
            n_users=n_users,
            n_items=n_items,
            n_personas=n_personas,
            lr=0.01,
            lamda=0.01,
            emb_dim=4,
            layer=layer,
            sparse_graph=sparse_graph,
            optimization="Adam",
            layer_weight_scheme="decay",
        )

        expected = [1 / (layer_idx + 1) for layer_idx in range(layer + 1)]
        assert model.layer_weight == pytest.approx(expected)

    def test_unknown_layer_weight_scheme_raises(self):
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        with pytest.raises(ValueError):
            model_LightGCN_tri(
                n_users=n_users,
                n_items=n_items,
                n_personas=n_personas,
                lr=0.01,
                lamda=0.01,
                emb_dim=4,
                layer=1,
                sparse_graph=sparse_graph,
                optimization="Adam",
                layer_weight_scheme="not_a_scheme",
            )

    def test_unknown_optimization_raises(self):
        n_users, n_items, n_personas = 3, 3, 2
        sparse_graph = make_identity_sparse_graph(n_users + n_items + n_personas)
        with pytest.raises(ValueError):
            model_LightGCN_tri(
                n_users=n_users,
                n_items=n_items,
                n_personas=n_personas,
                lr=0.01,
                lamda=0.01,
                emb_dim=4,
                layer=1,
                sparse_graph=sparse_graph,
                optimization="NotAnOptimizer",
            )
