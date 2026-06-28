"""
GPLR 논문 Appendix B의 3단계 페르소나 생성 프로세스를 퍼널 데이터셋에 적용.

Step 1 (40회): 랜덤 100명 구매이력 → LLM → 20개 후보
Step 2 (8회):  5세트(100개) 샘플링  → LLM → 정제된 20개
Step 3 (1회):  8세트(160개)         → LLM → 최종 20개 페르소나

실행: python3 funnel_persona_generation.py
결과: ../data/funnel_persona_gen/ 폴더에 저장
"""

import json
import os
import random
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI  # Upstage Solar API는 OpenAI SDK와 호환
from tqdm import tqdm

import datasets.MBA as mba_ds
import llm_connector.Collector as collector

warnings.filterwarnings("ignore")

DS_PATH = "../data/interim/funnel_mba_format.csv"
OUTPUT_DIR = "../data/interim/funnel_persona_gen/"

# ── 프롬프트 ──────────────────────────────────────────────────────────────────

STEP1_SYS = (
    "You are an assistant skilled at summarizing, capable of deducing "
    "high-level consumer keywords based on a user's purchases."
)

STEP1_USER = """\
Take a deep breath and work according to the instructions step by step.
Now you will conduct a series of analyses on a Funnel E-commerce dataset. \
This dataset contains data from an online retailer where each user's purchasing \
transactions and the bought items are recorded. I will provide purchase information \
from about 100 users; each user's data is grouped by ID and described in natural language.

Your task is to generate 20 representative and accurate user personas based on these \
users' purchasing patterns. Optimize for two targets:

• High Coverage: The persona set should cover as many users as possible. \
  Coverage = number of users that can be labeled with at least one persona.
• High Accuracy: Each persona must have a precise, non-ambiguous definition.

For each persona, provide a name and a concise one-sentence definition. \
Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
2. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

User purchasing data:
"""

STEP2_SYS = (
    "You are an assistant skilled at reading, observing and summarizing, capable of "
    "finding similar or repeated descriptions of user personas, and good at finding "
    "the most representative ones."
)

STEP2_USER = """\
Take a deep breath and work according to the instructions step by step.
I will give you 5 persona sets (each containing 20 personas), totaling 100 personas. \
Select the 20 most representative personas from these 100. \
If a set contains fewer than 20 personas or irrelevant information, ignore that set. \
You may find similar or duplicate personas — merge or pick the most representative. \
Do NOT use occurrence count as a criterion; judge solely by the persona descriptions.

Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

Here are the five persona sets:
"""

STEP3_SYS = STEP2_SYS

STEP3_USER = """\
Take a deep breath and work according to the instructions step by step.
I will give you 8 persona sets (each containing 20 personas), totaling 160 personas. \
Select the 20 most representative and distinct final personas from these 160. \
Prefer personas that appear most frequently across sets and cover the broadest user base.

Output exactly 20 personas in this format:
1. [Persona Name] - [Definition]
...
20. [Persona Name] - [Definition]

Here are the eight persona sets:
"""

# ── 유틸 ──────────────────────────────────────────────────────────────────────


def call_llm(client, sys_prompt, user_prompt, model="solar-pro"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def parse_persona_list(text):
    """'1. Name - Definition' 형태에서 {name, definition} 리스트 추출"""
    personas = []
    for line in text.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+?)[\s]*[-:]\s+(.+)$", line)
        if m:
            personas.append(
                {
                    "name": m.group(1).strip(),
                    "definition": m.group(2).strip(),
                }
            )
    return personas


def sample_user_descriptions(grouped_df, user_ids, n=100, seed=None):
    """n명 랜덤 샘플의 구매이력을 자연어로 변환"""
    if seed is not None:
        random.seed(seed)
    sampled = random.sample(list(user_ids), min(n, len(user_ids)))
    descs = [collector.describe_user(uid, grouped_df) for uid in sampled]
    return "\n\n".join(descs)


def format_persona_sets(persona_sets):
    """여러 세트를 번호 붙인 문자열로 변환"""
    blocks = []
    for i, ps in enumerate(persona_sets, 1):
        lines = [f"[Set {i}]"]
        for j, p in enumerate(ps, 1):
            lines.append(f"  {j}. {p['name']} - {p['definition']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ── 3단계 ─────────────────────────────────────────────────────────────────────


def step1(client, grouped_df, user_ids, n_iter=40, users_per_iter=100, workers=8):
    """Step 1: 40회 반복, 각 100명 샘플로 20개 후보 생성"""
    print(f"\n[Step 1] {n_iter}회 반복 실행 중 (모델당 100명 샘플)...")

    def single_run(i):
        user_data = sample_user_descriptions(grouped_df, user_ids, n=users_per_iter, seed=i * 7)
        response = call_llm(client, STEP1_SYS, STEP1_USER + user_data)
        personas = parse_persona_list(response)
        return {"iter": i, "personas": personas, "raw": response}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(single_run, i) for i in range(n_iter)]
        for f in tqdm(as_completed(futures), total=n_iter, desc="Step 1"):
            results.append(f.result())

    results.sort(key=lambda x: x["iter"])
    return results


def step2(client, step1_results, n_iter=8, sets_per_sample=5, workers=4):
    """Step 2: 40세트 중 5세트씩 샘플링, 8회 반복으로 정제"""
    print(f"\n[Step 2] {n_iter}회 반복 실행 중 (5세트씩 샘플링)...")
    all_sets = [r["personas"] for r in step1_results]

    def single_run(i):
        random.seed(i * 13)
        sampled = random.sample(all_sets, sets_per_sample)
        context = format_persona_sets(sampled)
        response = call_llm(client, STEP2_SYS, STEP2_USER + context)
        personas = parse_persona_list(response)
        return {"iter": i, "personas": personas, "raw": response}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(single_run, i) for i in range(n_iter)]
        for f in tqdm(as_completed(futures), total=n_iter, desc="Step 2"):
            results.append(f.result())

    results.sort(key=lambda x: x["iter"])
    return results


def step3(client, step2_results):
    """Step 3: 8세트(160개) → 최종 20개 페르소나"""
    print("\n[Step 3] 최종 20개 페르소나 확정 중...")
    all_sets = [r["personas"] for r in step2_results]
    context = format_persona_sets(all_sets)
    response = call_llm(client, STEP3_SYS, STEP3_USER + context)
    personas = parse_persona_list(response)
    return personas, response


# ── 메인 ──────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("데이터 로딩 중...")
    [mba_df, user_ids, user_num, *_] = mba_ds.MBA_load_data(DS_PATH)
    grouped_df = mba_df.groupby(["CustomerID", "Itemname"]).agg({"Quantity": "sum"}).reset_index()

    client = OpenAI(
        api_key=os.environ.get("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
    )

    # Step 1
    s1 = step1(client, grouped_df, user_ids)
    with open(OUTPUT_DIR + "step1_persona_sets.json", "w", encoding="utf-8") as f:
        json.dump(s1, f, ensure_ascii=False, indent=2)
    print(f"→ Step 1 저장: {OUTPUT_DIR}step1_persona_sets.json")

    # Step 2
    s2 = step2(client, s1)
    with open(OUTPUT_DIR + "step2_persona_sets.json", "w", encoding="utf-8") as f:
        json.dump(s2, f, ensure_ascii=False, indent=2)
    print(f"→ Step 2 저장: {OUTPUT_DIR}step2_persona_sets.json")

    # Step 3
    final_personas, raw = step3(client, s2)
    with open(OUTPUT_DIR + "final_personas.json", "w", encoding="utf-8") as f:
        json.dump(final_personas, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR + "final_personas_raw.txt", "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"→ Step 3 저장: {OUTPUT_DIR}final_personas.json")

    print(f"\n=== 최종 페르소나 목록 ({len(final_personas)}개) ===")
    for i, p in enumerate(final_personas, 1):
        print(f"{i:2d}. {p['name']}")
        print(f"     {p['definition']}")


if __name__ == "__main__":
    main()
