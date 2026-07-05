"""ID 인코딩 공용 유틸리티.

user_id/item_id처럼 원본 ID를 그래프·행렬 인덱스로 쓰기 위해 0부터 시작하는
정수로 변환한다. ALS(als_model.py)와 LightGCN(build_tri_graph_data.py) 양쪽에서
동일한 방식으로 재사용한다.
"""

import pandas as pd


def build_id_encoding(ids: pd.Series) -> tuple[dict, dict]:
    """고유 ID 목록을 0부터 시작하는 정수 인덱스로 인코딩한다.

    반환값: (encoding: {원본 id -> 인덱스}, decoding: {인덱스 -> 원본 id})
    """
    unique_ids = ids.unique()
    encoding = {original_id: idx for idx, original_id in enumerate(unique_ids)}
    decoding = {idx: original_id for original_id, idx in encoding.items()}
    return encoding, decoding
