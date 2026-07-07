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
            top_items = sess.run(
                model.top_items,
                feed_dict={
                    model.users: [0, 1],
                    model.items_in_train_data: np.zeros((2, n_items), dtype=np.float32),
                    model.top_k: 2,
                },
            )
            assert top_items.shape == (2, 2)

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
