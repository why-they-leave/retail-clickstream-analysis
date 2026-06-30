"""
생성된 20개 페르소나를 기반으로 유저/상품에 페르소나 라벨 부여.

실행: python3 labeling.py
결과: ../data/funnel_persona_gen/ 에 저장
  - persona2idx.json
  - tri_graph_uidx2tidx_train.json
  - tri_graph_uidx2tidx_test.json
  - tri_graph_uidx2pidx.json   (유저 → 페르소나)
  - tri_graph_tidx2pidx.json   (상품 → 페르소나)
"""

import json
import logging
import os
import random
import re
import warnings
from argparse import ArgumentParser

import pandas as pd
from openai import OpenAI

import datasets.MBA as mba_ds
from llm_connector.client import call_llm, run_parallel
from llm_connector.env import get_required_env
from llm_connector.formatter import describe_user
from llm_connector.parser import parse_item_response, parse_user_response
from persona.config import load_persona_config

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ── 페르소나 로딩 ──────────────────────────────────────────────────────────────


def load_personas(path):
    """final_personas.json 로드 후 ** 마크다운 제거."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        {"name": re.sub(r"\*+", "", p["name"]).strip(), "definition": p["definition"]} for p in raw
    ]


def build_persona2idx(personas):
    """페르소나명 → 인덱스 매핑 생성."""
    return {p["name"]: i for i, p in enumerate(personas)}


def format_persona_list_for_prompt(personas):
    """페르소나 목록을 프롬프트용 텍스트로 변환."""
    return "\n".join(
        [f"    {i}.  {p['name']} - {p['definition']}" for i, p in enumerate(personas, 1)]
    )


# ── 프롬프트 ──────────────────────────────────────────────────────────────────


def make_user_prompts(personas):
    """유저 라벨링용 시스템/유저 프롬프트 생성."""
    persona_block = format_persona_list_for_prompt(personas)
    sys_prompt = (
        "Now you are an intelligent e-commerce domain assistant. "
        "You are skilled at summarizing, and capable of assigning high-level "
        "consumer personas based on a user's purchase behavior."
    )
    user_prompt = f"""Take a deep breath and work according to the instructions step by step.
Your goal is to identify users' shopping behaviors based on products they have bought \
and label them with a given set of personas. Select at least one persona, at most 5 personas \
from the given list. Make sure each assignment has strong evidence in their purchase transactions.

Select only exact persona names from the given list. \
Do not rename, paraphrase, abbreviate, or create new personas.

Only output valid JSON. No markdown. No explanation.

Output format:
{{ "user_number": ["Persona1", "Persona2"] }}

Example:
{{ "1001": ["Tech & Gadget Shopper", "Frequent Repurchaser"] }}

If no persona fits, label as Unrepresentable:
{{ "9999": ["Unrepresentable"] }}

Here is the persona list you should choose from:
{persona_block}

Remember: the user number must match exactly from the transaction data.
"""
    return sys_prompt, user_prompt


def make_item_prompts(personas):
    """상품 라벨링용 시스템/유저 프롬프트 생성."""
    persona_block = format_persona_list_for_prompt(personas)
    sys_prompt = (
        "Now you are an intelligent e-commerce domain assistant. "
        "You have a good understanding of various customer personas, "
        "and are skilled at connecting products with personas most likely to be interested in them."
    )
    user_prompt = f"""Take a deep breath and work according to the instructions step by step.

I will provide a product name and a set of customer personas. \
Your task is to connect the product with a subset of personas that would likely purchase it.

Principles:
1. Understand each persona strictly by its definition.
2. Use real-world knowledge to understand the product's attributes from its name.
3. Only connect personas with strong, well-substantiated relevance to the product.

Select only exact persona names from the given list. \
Do not rename, paraphrase, abbreviate, or create new personas.

Only output valid JSON. No markdown. No explanation.

Personas:
{persona_block}

Output format: a JSON dict mapping the exact product name to a list of 1–5 relevant persona names.
Example: {{"SSD MediumBlue 149": ["Tech & Gadget Shopper", "Single-Category Specialist"]}}

Product name: """
    return sys_prompt, user_prompt


# ── LLM 호출 단위 ─────────────────────────────────────────────────────────────


def assign_user_label(
    client,
    sys_prompt,
    user_prompt,
    uid,
    grouped_df,
    train_items,
    model,
    customer_features=None,
):
    """유저 한 명에 대해 LLM 라벨링 호출."""
    train_item_names = [item_names[t] for t in train_items]
    transaction_data = describe_user(uid, grouped_df, train_item_names, customer_features)
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


def assign_item_label(client, sys_prompt, user_prompt, itemname, model):
    """상품 하나에 대해 LLM 라벨링 호출."""
    try:
        response = call_llm(client, sys_prompt, user_prompt + itemname, model=model)
        return {itemname: response}
    except Exception as e:
        logger.error("item %s: %s", itemname, e)
        return {itemname: "QUERY_FAILED"}


def handle_user_task_error(error, task_args):
    """병렬 유저 라벨링 task 예외를 실패 결과로 변환."""
    uid = task_args[3]
    logger.error("user %s: %s", uid, error)
    return {"user": uid, "profile": "QUERY_FAILED"}


def handle_item_task_error(error, task_args):
    """병렬 상품 라벨링 task 예외를 실패 결과로 변환."""
    itemname = task_args[3]
    logger.error("item %s: %s", itemname, error)
    return {itemname: "QUERY_FAILED"}


# ── 메인 ──────────────────────────────────────────────────────────────────────


def parse_args():
    parser = ArgumentParser(description="Label users and items with generated personas.")
    parser.add_argument("--config", default=None, help="Path to persona config YAML.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_persona_config(args.config)
    paths = config["paths"]
    llm = config["llm"]
    sample = config["sample"]
    persona_cfg = config["persona"]

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로딩
    logger.info("데이터 로딩 중...")
    global item_names  # assign_user_label에서 사용
    [mba_df, user_ids, user_num, user_ids_kv, item_names, item_num, items_kv, G_user, G_item] = (
        mba_ds.MBA_load_data(str(paths["input"]), country=sample.get("country"))
    )
    grouped_df = mba_df.groupby(["CustomerID", "Itemname"]).agg({"Quantity": "sum"}).reset_index()

    # 집계 피처 로딩
    customer_features = None
    customer_features_path = paths["customer_features"]
    if os.path.exists(customer_features_path):
        customer_features = pd.read_csv(customer_features_path)
        logger.info("집계 피처 로드 완료: %d명", len(customer_features))
    else:
        logger.warning("집계 피처 파일 없음: %s — 기본 프롬프트로 실행", customer_features_path)

    # 페르소나 로딩
    personas = load_personas(paths["persona"])
    persona2idx = build_persona2idx(personas)
    defined_persona_set = set(persona2idx.keys()) | {"Unrepresentable"}
    logger.info("페르소나 %d개 로드 완료", len(personas))

    with open(output_dir / "persona2idx.json", "w", encoding="utf-8") as f:
        json.dump(persona2idx, f, ensure_ascii=False, indent=2)

    # Train / Test 분할 (80 / 20)
    # 날짜 기반이 아닌 per-user 아이템 shuffle split (stratify 없음)
    random_state = int(sample["random_state"])
    train_ratio = float(persona_cfg["train_ratio"])
    random.seed(random_state)
    tri_graph_uidx2tidx_train, tri_graph_uidx2tidx_test = {}, {}
    for uidx in G_user:
        items = list(G_user[uidx])
        random.shuffle(items)
        n_items = len(items)
        ltrain = max(int(train_ratio * n_items), 1)
        tri_graph_uidx2tidx_train[uidx] = items[:ltrain]
        tri_graph_uidx2tidx_test[uidx] = items[ltrain:]

    with open(output_dir / "tri_graph_uidx2tidx_train.json", "w") as f:
        json.dump(tri_graph_uidx2tidx_train, f)
    with open(output_dir / "tri_graph_uidx2tidx_test.json", "w") as f:
        json.dump(tri_graph_uidx2tidx_test, f)
    logger.info("Train/Test 분할 저장 완료")

    client = OpenAI(
        api_key=get_required_env("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
    )
    sys_u, usr_u = make_user_prompts(personas)
    sys_i, usr_i = make_item_prompts(personas)
    model = llm["model"]
    workers = int(llm["max_workers"])

    # ── 유저 샘플링 ──────────────────────────────────────────────────────────
    sample_rate = float(sample["ratio"])
    sample_size = max(1, int(user_num * sample_rate))
    random.seed(random_state)
    sampled_uidxs = random.sample(range(user_num), sample_size)
    logger.info(
        "[유저 샘플링] 전체 %d명 중 %d명 (%.1f%%) 추출", user_num, sample_size, sample_rate * 100
    )

    # ── 유저 라벨링 (병렬) ───────────────────────────────────────────────────
    logger.info("[유저 라벨링] %d명 처리 중...", sample_size)
    tasks_args = [
        (
            client,
            sys_u,
            usr_u,
            user_ids[uidx],
            grouped_df,
            tri_graph_uidx2tidx_train[uidx],
            model,
            customer_features,
        )
        for uidx in sampled_uidxs
    ]
    user_results = run_parallel(
        assign_user_label,
        tasks_args,
        max_workers=workers,
        desc="유저",
        on_error=handle_user_task_error,
    )

    tri_graph_uidx2pidx = {}
    fail_count = 0
    for r in user_results:
        uid = r["user"]
        profile = r["profile"]
        if profile == "QUERY_FAILED":
            fail_count += 1
            continue
        parsed = parse_user_response(uid, profile, defined_persona_set)
        if parsed is None:
            fail_count += 1
            continue
        ps = list(parsed.values())[0]
        if "Unrepresentable" in ps:
            continue
        uidx = user_ids_kv[uid]
        ps_idxs = [persona2idx[p] for p in ps if p in persona2idx]
        if ps_idxs:
            tri_graph_uidx2pidx[uidx] = ps_idxs

    logger.info("유저 라벨링 완료: %d명 / 실패 %d명", len(tri_graph_uidx2pidx), fail_count)
    with open(output_dir / "tri_graph_uidx2pidx.json", "w") as f:
        json.dump({int(k): v for k, v in tri_graph_uidx2pidx.items()}, f)

    # ── 상품 라벨링 (병렬) ───────────────────────────────────────────────────
    logger.info("[상품 라벨링] %d개 처리 중...", item_num)
    tasks_args_items = [(client, sys_i, usr_i, name, model) for name in item_names]
    item_results = run_parallel(
        assign_item_label,
        tasks_args_items,
        max_workers=workers,
        desc="상품",
        on_error=handle_item_task_error,
    )

    tri_graph_tidx2pidx = {}
    for r in item_results:
        itemname = list(r.keys())[0]
        response = list(r.values())[0]
        if response == "QUERY_FAILED":
            continue
        ps = parse_item_response(response, itemname, defined_persona_set)
        if ps:
            tidx = items_kv[itemname]
            tri_graph_tidx2pidx[tidx] = [persona2idx[p] for p in ps]

    logger.info("상품 라벨링 완료: %d개", len(tri_graph_tidx2pidx))
    with open(output_dir / "tri_graph_tidx2pidx.json", "w") as f:
        json.dump({int(k): v for k, v in tri_graph_tidx2pidx.items()}, f)

    logger.info("=== 모든 라벨링 완료 === 저장 위치: %s", output_dir)


if __name__ == "__main__":
    main()
