"""
v1 taxonomy 기준 유저 라벨링만 실행한다 (Issue #18 — v1 vs v2 세그먼트 품질 비교용).

labeling.py는 유저 라벨링과 상품 라벨링을 모두 실행하지만, #18의 비교 지표
(coverage, entropy, top-k concentration, segment size balance)는 유저→persona
배정 결과만 있으면 계산할 수 있다. 상품 라벨링(전체 상품 1,195개 전수 호출)은
추천시스템 통합용 산출물이라 이 단계에서는 생략해 불필요한 LLM 호출을 없앤다.

labeling.py와의 차이:
    - 상품 라벨링 없음
    - train/test 아이템 분할 없음 — 고객의 전체 구매이력을 그대로 설명에 사용
      (v2 segment_summary도 고객 전체 행동 데이터 기준이라 비교 조건을 맞춤)

실행 (cwd: src/):
    python3.11 -m persona.label_users_only [--config PATH]

결과:
    data/interim/funnel_persona_gen/user_persona_labels_v1.json
"""

import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from openai import OpenAI

import datasets.MBA as mba_ds
from llm_connector.client import call_llm, run_parallel
from llm_connector.env import get_required_env
from llm_connector.formatter import describe_user
from llm_connector.parser import parse_user_response
from persona.config import load_persona_config
from persona.labeling import (
    build_persona2idx,
    handle_user_task_error,
    load_personas,
    make_user_prompts,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILENAME = "user_persona_labels_v1.json"


def label_one_user(client, sys_prompt, user_prompt, uid, grouped_df, model, customer_features=None):
    """유저 한 명에 대해 LLM 라벨링 호출. 전체 구매이력을 사용한다(train/test 분할 없음)."""
    transaction_data = describe_user(uid, grouped_df, None, customer_features)
    tail = (
        f"\nHere is user {uid}'s transaction data:\n{transaction_data}\n"
        "Select only from the given persona list. "
        "Only output valid JSON. No markdown. No explanation."
    )
    try:
        response = call_llm(client, sys_prompt, user_prompt + tail, model=model)
        return {"user": uid, "profile": response}
    except Exception as e:
        logger.error("user %s: %s", uid, e)
        return {"user": uid, "profile": "QUERY_FAILED"}


def parse_args():
    parser = ArgumentParser(
        description="v1 유저 라벨링만 실행 (Issue #18 비교용, 상품 라벨링 생략)."
    )
    parser.add_argument("--config", default=None, help="Path to persona config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_persona_config(args.config)
    paths = config["paths"]
    llm = config["llm"]
    sample = config["sample"]

    output_dir: Path = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("데이터 로딩 중...")
    [mba_df, user_ids, user_num, user_ids_kv, item_names, item_num, items_kv, G_user, G_item] = (
        mba_ds.MBA_load_data(str(paths["input"]), country=sample.get("country"))
    )
    grouped_df = mba_df.groupby(["CustomerID", "Itemname"]).agg({"Quantity": "sum"}).reset_index()

    customer_features = None
    if paths["customer_features"].exists():
        customer_features = pd.read_csv(paths["customer_features"])
        logger.info("집계 피처 로드 완료: %d명", len(customer_features))
    else:
        logger.warning("집계 피처 파일 없음: %s — 기본 프롬프트로 실행", paths["customer_features"])

    personas = load_personas(paths["persona"])
    persona2idx = build_persona2idx(personas)
    defined_persona_set = set(persona2idx.keys()) | {"Unrepresentable"}
    logger.info("페르소나 %d개 로드 완료: %s", len(personas), paths["persona"])

    sys_u, usr_u = make_user_prompts(personas)
    model = llm["model"]
    workers = int(llm["max_workers"])

    sample_rate = float(sample["ratio"])
    random_state = int(sample["random_state"])
    sample_size = max(1, int(user_num * sample_rate))
    random.seed(random_state)
    sampled_uidxs = random.sample(range(user_num), sample_size)
    logger.info(
        "[유저 샘플링] 전체 %d명 중 %d명 (%.1f%%) 추출", user_num, sample_size, sample_rate * 100
    )

    client = OpenAI(
        api_key=get_required_env("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
    )

    tasks_args = [
        (client, sys_u, usr_u, user_ids[uidx], grouped_df, model, customer_features)
        for uidx in sampled_uidxs
    ]
    logger.info("[유저 라벨링] %d명 처리 중...", sample_size)
    user_results = run_parallel(
        label_one_user,
        tasks_args,
        max_workers=workers,
        desc="유저",
        on_error=handle_user_task_error,
    )

    labels = []
    for r in user_results:
        uid = r["user"]
        profile = r["profile"]
        if profile == "QUERY_FAILED":
            labels.append({"customer_id": int(uid), "personas": [], "status": "query_failed"})
            continue

        parsed = parse_user_response(uid, profile, defined_persona_set)
        if parsed is None:
            labels.append({"customer_id": int(uid), "personas": [], "status": "unparseable"})
            continue

        ps = list(parsed.values())[0]
        if "Unrepresentable" in ps:
            labels.append({"customer_id": int(uid), "personas": [], "status": "unrepresentable"})
            continue

        labels.append({"customer_id": int(uid), "personas": ps, "status": "ok"})

    status_counts = {}
    for label in labels:
        status_counts[label["status"]] = status_counts.get(label["status"], 0) + 1
    logger.info("[유저 라벨링] 완료: %s", status_counts)

    output = {
        "sample_size": sample_size,
        "total_users": user_num,
        "sample_ratio": sample_rate,
        "random_state": random_state,
        "persona_source": str(paths["persona"]),
        "labels": labels,
    }
    output_path = output_dir / OUTPUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("저장: %s", output_path)


if __name__ == "__main__":
    main()
