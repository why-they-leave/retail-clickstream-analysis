"""
생성된 20개 페르소나를 기반으로 유저/상품에 페르소나 라벨 부여.

실행: python3 funnel_persona_labeling.py
결과: ../data/funnel_persona_gen/ 에 저장
  - persona2idx.json
  - tri_graph_uidx2tidx_train.json
  - tri_graph_uidx2tidx_test.json
  - tri_graph_uidx2pidx.json   (유저 → 페르소나)
  - tri_graph_tidx2pidx.json   (상품 → 페르소나)
"""

import json
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

import datasets.MBA as mba_ds
import llm_connector.Collector as collector

warnings.filterwarnings("ignore")

DS_PATH = "../data/funnel_mba_format.csv"
PERSONA_PATH = "../data/funnel_persona_gen/final_personas.json"
OUTPUT_DIR = "../data/funnel_persona_gen/"
MODEL = "solar-pro"

# ── 페르소나 로딩 ──────────────────────────────────────────────────────────────


def load_personas(path):
    """final_personas.json 로드 후 ** 마크다운 제거"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    personas = []
    for p in raw:
        name = re.sub(r"\*+", "", p["name"]).strip()
        personas.append({"name": name, "definition": p["definition"]})
    return personas


def build_persona2idx(personas):
    return {p["name"]: i for i, p in enumerate(personas)}


def format_persona_list_for_prompt(personas):
    lines = []
    for i, p in enumerate(personas, 1):
        lines.append(f"    {i}.  {p['name']} - {p['definition']}")
    return "\n".join(lines)


# ── 프롬프트 ──────────────────────────────────────────────────────────────────


def make_user_prompts(personas):
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

Please provide the output in json format:
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

Personas:
{persona_block}

Output: a JSON dict mapping the exact product name to a list of 1–5 relevant persona names.
Example: {{"SSD MediumBlue 149": ["Tech & Gadget Shopper", "Single-Category Specialist"]}}

You do not need to explain your reasoning. Only output the JSON.

Product name: """
    return sys_prompt, user_prompt


# ── LLM 호출 ──────────────────────────────────────────────────────────────────


def call_llm(client, sys_prompt, user_prompt):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def assign_user_label(client, sys_prompt, user_prompt, uid, grouped_df, train_items):
    train_item_names = [item_names[t] for t in train_items]
    transaction_data = collector.describe_user(uid, grouped_df, train_item_names)
    tail = (
        f"\nHere is user {uid}'s transaction data:\n{transaction_data}\n"
        "Select only from the given persona list. Do not explain — only output the JSON."
    )
    try:
        response = call_llm(client, sys_prompt, user_prompt + tail)
        return {"user": uid, "profile": response}
    except Exception as e:
        print(f"[E] user {uid}: {e}")
        return {"user": uid, "profile": "QUERY_FAILED"}


def assign_item_label(client, sys_prompt, user_prompt, itemname):
    try:
        response = call_llm(client, sys_prompt, user_prompt + itemname)
        return {itemname: response}
    except Exception as e:
        print(f"[E] item {itemname}: {e}")
        return {itemname: "QUERY_FAILED"}


# ── 파싱 ──────────────────────────────────────────────────────────────────────


def parse_user_response(uid, answer_str, defined_persona_set):
    """LLM 응답에서 {uid: [persona_names]} 추출"""
    uid_str = str(uid)
    # JSON 블록 추출
    start = answer_str.find("{")
    end = answer_str.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        res = json.loads(answer_str[start:end])
    except Exception:
        try:
            res = eval(answer_str[start:end])
        except Exception:
            return None

    # uid 키 탐색 (int or str)
    val = res.get(uid_str) or res.get(uid) or res.get(int(uid_str) if uid_str.isdigit() else uid)
    if val is None:
        return None

    # 유효한 페르소나만 필터링
    valid = [p for p in val if p in defined_persona_set or p == "Unrepresentable"]
    return {uid_str: valid} if valid else None


def parse_item_response(answer_str, itemname, defined_persona_set):
    """LLM 응답에서 {itemname: [persona_names]} 추출"""
    start = answer_str.find("{")
    end = answer_str.rfind("}") + 1
    if start == -1 or end == 0:
        return []
    try:
        res = json.loads(answer_str[start:end])
    except Exception:
        try:
            res = eval(answer_str[start:end])
        except Exception:
            return []
    val = list(res.values())[0] if res else []
    return [p for p in val if p in defined_persona_set]


# ── 메인 ──────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 데이터 로딩
    print("데이터 로딩 중...")
    global item_names  # assign_user_label에서 사용
    [mba_df, user_ids, user_num, user_ids_kv, item_names, item_num, items_kv, G_user, G_item] = (
        mba_ds.MBA_load_data(DS_PATH)
    )
    grouped_df = mba_df.groupby(["CustomerID", "Itemname"]).agg({"Quantity": "sum"}).reset_index()

    # 페르소나 로딩
    personas = load_personas(PERSONA_PATH)
    persona2idx = build_persona2idx(personas)
    defined_persona_set = set(persona2idx.keys()) | {"Unrepresentable"}
    print(f"페르소나 {len(personas)}개 로드 완료")

    # persona2idx 저장
    with open(OUTPUT_DIR + "persona2idx.json", "w", encoding="utf-8") as f:
        json.dump(persona2idx, f, ensure_ascii=False, indent=2)

    # Train / Test 분할 (80 / 20)
    random.seed(191)
    tri_graph_uidx2tidx_train, tri_graph_uidx2tidx_test = {}, {}
    for uidx in G_user:
        items = list(G_user[uidx])
        random.shuffle(items)
        n_items = len(items)
        ltrain = max(int(0.8 * n_items), 1)
        tri_graph_uidx2tidx_train[uidx] = items[:ltrain]
        tri_graph_uidx2tidx_test[uidx] = items[ltrain:]

    with open(OUTPUT_DIR + "tri_graph_uidx2tidx_train.json", "w") as f:
        json.dump(tri_graph_uidx2tidx_train, f)
    with open(OUTPUT_DIR + "tri_graph_uidx2tidx_test.json", "w") as f:
        json.dump(tri_graph_uidx2tidx_test, f)
    print("Train/Test 분할 저장 완료")

    client = OpenAI(
        api_key=os.environ.get("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
    )
    sys_u, usr_u = make_user_prompts(personas)
    sys_i, usr_i = make_item_prompts(personas)

    # ── 유저 샘플링 (전체의 2.5%) ────────────────────────────────────────────
    sample_rate = 0.025
    sample_size = max(1, int(user_num * sample_rate))
    random.seed(191)
    sampled_uidxs = random.sample(range(user_num), sample_size)
    print(f"\n[유저 샘플링] 전체 {user_num:,}명 중 {sample_size}명 ({sample_rate * 100:.1f}%) 추출")

    # ── 유저 라벨링 ──────────────────────────────────────────────────────────
    print(f"[유저 라벨링] {sample_size}명 처리 중...")

    user_results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [
            ex.submit(
                assign_user_label,
                client,
                sys_u,
                usr_u,
                user_ids[uidx],
                grouped_df,
                tri_graph_uidx2tidx_train[uidx],
            )
            for uidx in sampled_uidxs
        ]
        for f in tqdm(as_completed(futures), total=sample_size, desc="유저"):
            user_results.append(f.result())

    # 파싱 및 저장
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

    print(f"유저 라벨링 완료: {len(tri_graph_uidx2pidx):,}명 / 실패 {fail_count}명")
    with open(OUTPUT_DIR + "tri_graph_uidx2pidx.json", "w") as f:
        json.dump({int(k): v for k, v in tri_graph_uidx2pidx.items()}, f)

    # ── 상품 라벨링 ──────────────────────────────────────────────────────────
    print(f"\n[상품 라벨링] {item_num:,}개 처리 중...")

    item_results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(assign_item_label, client, sys_i, usr_i, name) for name in item_names]
        for f in tqdm(as_completed(futures), total=item_num, desc="상품"):
            item_results.append(f.result())

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

    print(f"상품 라벨링 완료: {len(tri_graph_tidx2pidx):,}개")
    with open(OUTPUT_DIR + "tri_graph_tidx2pidx.json", "w") as f:
        json.dump({int(k): v for k, v in tri_graph_tidx2pidx.items()}, f)

    print("\n=== 모든 라벨링 완료 ===")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
