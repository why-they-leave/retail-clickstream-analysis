"""
Segment summary 기반 LLM naming/summarization (Issue #17).

v1(persona/generation.py, persona/labeling.py)과 달리 LLM은 persona taxonomy를
만들거나 개별 고객을 배정하지 않는다. Issue #16이 이미 확정한 segment_id별
집계 통계(segment_summary_*.csv)만 근거로 사용해, 사람이 읽을 수 있는
이름/설명을 붙이는 라벨러 역할로 제한한다.

Full 데이터 단일 트랙 (Issue #23 — US-only 트랙 제거).

입력:
    data/processed/segment_summary_all_customers.csv

실험/채택 흐름:
    매 실행은 experiments/segment_naming_v2/run_<날짜>_<순번>/segment_personas.json에
    저장된다 (canonical 파일은 건드리지 않음). 후보들 중 evidence가 segment_summary
    수치와 정확히 일치하고 demographic/lifestyle 표현이 없는 run을 사람이 검토해
    experiments/segment_naming_v2/CHOICES.md에 채택 사유를 기록한 뒤,
    --promote RUN_LABEL로 canonical 파일에 반영한다.

출력:
    data/processed/segment_personas_v2.json
"""

import json
import logging
import shutil
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

from llm_connector.client import call_llm
from llm_connector.env import get_required_env
from llm_connector.parser import _parse_json_block

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXPERIMENTS_DIR = ROOT_DIR / "experiments" / "segment_naming_v2"

SEGMENT_SUMMARY_PATH = PROCESSED_DIR / "segment_summary_all_customers.csv"
OUTPUT_PATH = PROCESSED_DIR / "segment_personas_v2.json"
CUSTOMER_SEGMENTS_PATH = PROCESSED_DIR / "customer_segments_all_customers.csv"
LABELED_OUTPUT_PATH = PROCESSED_DIR / "customer_segments_labeled_all_customers.csv"

REQUIRED_KEYS = ["segment_id", "segment_name", "description", "evidence", "cautions"]

# 상수 추가
PERSONA_MERGE_COLUMNS = [
    "segment_id",
    "segment_name",
    "description",
    "evidence",
    "cautions",
    "status",
    "errors",
]

# 하드 실패 기준은 아니고, 결과에 남겨 사람이 검토하도록 하는 소프트 경고 목록이다.
# "Single-Category Specialist"처럼 행동 기반 표현에도 등장하는 단어(single 등)가
# 있어 자동 거부는 오탐이 많다 — 완료 기준은 "prompt에 금지 규칙이 포함되는 것"이지
# 완벽한 자동 차단이 아니다.
DEMOGRAPHIC_KEYWORDS = [
    "young",
    "senior",
    "teen",
    "elderly",
    "urban",
    "rural",
    "suburban",
    "affluent",
    "wealthy",
    "budget-conscious family",
    "family",
    "parent",
    "professional",
    "student",
    "executive",
    "income",
    "occupation",
]


def _fmt_pct(value, decimals: int = 1) -> str:
    """비율 값을 퍼센트 문자열로 변환. 결측이면 'none'."""
    if pd.isna(value):
        return "none"
    return f"{value * 100:.{decimals}f}%"


def _fmt_num(value, decimals: int = 2, suffix: str = "") -> str:
    """실수 값을 고정 소수점 문자열로 변환. 결측이면 'none'."""
    if pd.isna(value):
        return "none"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_text(value) -> str:
    """카테고리 분포 등 텍스트 값. 결측/빈 문자열이면 'none'."""
    if pd.isna(value) or str(value).strip() == "":
        return "none"
    return str(value)


def format_segment_summary(row: pd.Series) -> str:
    """segment_summary_*.csv 한 row를 LLM naming 프롬프트용 텍스트로 변환한다."""
    segment_id = int(row["segment_id"])
    lines = [
        f"Segment {segment_id} summary:",
        f"- Customers: {int(row['customer_count']):,} ({_fmt_pct(row['customer_ratio'])} of all customers)",
        f"- Avg page views: {_fmt_num(row['avg_page_view_count'])}",
        f"- Avg add-to-cart count: {_fmt_num(row['avg_add_to_cart_count'])}",
        f"- Avg add-to-cart rate: {_fmt_pct(row['avg_atc_rate'])}",
        f"- Purchaser ratio: {_fmt_pct(row['purchaser_ratio'])}",
        f"- Avg order count: {_fmt_num(row['avg_order_count'])}",
        f"- Avg purchase per session: {_fmt_num(row['avg_purchase_per_session'])}",
        f"- Avg total spend (log-scale): {_fmt_num(row['avg_total_spend_log'])}",
        f"- Avg days since last session: {_fmt_num(row['avg_recency_session_days'], decimals=1)}",
        f"- Avg days since last order: {_fmt_num(row['avg_recency_order_days'], decimals=1)}",
        f"- Avg purchase category diversity: {_fmt_num(row['avg_category_diversity_purchase'])}",
        f"- Avg dominant view category ratio: {_fmt_pct(row['avg_dominant_view_category_ratio'])}",
        f"- Avg dominant purchase category ratio: {_fmt_pct(row['avg_dominant_purchase_category_ratio'])}",
        f"- View-purchase category match rate: {_fmt_pct(row['view_purchase_category_match_rate'])}",
        f"- Top view categories: {_fmt_text(row['top_view_categories'])}",
        f"- Top purchase categories: {_fmt_text(row['top_purchase_categories'])}",
    ]
    return "\n".join(lines)


SEGMENT_NAMING_SYS = (
    "You are naming customer behavior segments for an ecommerce analytics system. "
    "You act strictly as a labeler: you translate already-computed segment "
    "statistics into a human-readable name and description. You are not analyzing "
    "individual customers and you are not inventing a persona taxonomy."
)

SEGMENT_NAMING_USER_TEMPLATE = """\
Take a deep breath and work according to the instructions step by step.

{segment_summary}

Task:
Create a concise segment name and description using only the evidence above.

Rules:
- Use only the statistics given above. Do not use raw product names, individual \
customer records, or any data not shown in this summary.
- Do not infer income, age, occupation, family status, or lifestyle. These are not \
observable in the data.
- Prefer observable behavior-based phrasing (e.g. High-Intent, Repeat, \
Non-Purchasing, Browsing-Only) over demographic/lifestyle phrasing (e.g. Young, \
Urban, Affluent, Family, Professional).
- This segment was produced by KMeans clustering with generally low silhouette \
scores, so treat it as a behavior-based approximation, not a sharply separated \
natural group. Avoid overconfident or marketing-style labels.
- Every item in "evidence" must restate a specific field/value from the summary \
above. Do not add evidence that isn't in the summary.
- If a field above is "none", do not claim behavior about it.
- Return valid JSON only. No markdown. No explanation.

Output JSON schema:
{{
  "segment_id": <integer, must equal {segment_id}>,
  "segment_name": "<string>",
  "description": "<string>",
  "evidence": ["<string>", ...],
  "cautions": ["<string>", ...]
}}
"""


def build_user_prompt(row: pd.Series) -> str:
    """segment summary row로부터 naming user prompt를 완성한다."""
    return SEGMENT_NAMING_USER_TEMPLATE.format(
        segment_summary=format_segment_summary(row),
        segment_id=int(row["segment_id"]),
    )


def validate_naming_response(parsed, expected_segment_id: int) -> list[str]:
    """LLM 응답이 output JSON schema를 만족하는지 검증한다."""
    if not isinstance(parsed, dict):
        return ["응답이 JSON object가 아님"]

    missing = [key for key in REQUIRED_KEYS if key not in parsed]
    if missing:
        return [f"필수 키 누락: {missing}"]

    errors = []
    if parsed["segment_id"] != expected_segment_id:
        errors.append(
            f"segment_id 불일치: 기대값={expected_segment_id}, 응답값={parsed['segment_id']!r}"
        )
    if not isinstance(parsed["segment_name"], str) or not parsed["segment_name"].strip():
        errors.append("segment_name이 비어있거나 문자열이 아님")
    if not isinstance(parsed["description"], str) or not parsed["description"].strip():
        errors.append("description이 비어있거나 문자열이 아님")
    if not isinstance(parsed["evidence"], list) or not all(
        isinstance(e, str) for e in parsed["evidence"]
    ):
        errors.append("evidence가 문자열 리스트가 아님")
    if not isinstance(parsed["cautions"], list) or not all(
        isinstance(c, str) for c in parsed["cautions"]
    ):
        errors.append("cautions가 문자열 리스트가 아님")
    return errors


def check_demographic_language(parsed: dict) -> list[str]:
    """LLM 출력 문자열 필드에 demographic/lifestyle 표현이 있는지 소프트 체크한다."""
    fields = [parsed.get("segment_name", ""), parsed.get("description", "")]
    fields.extend(parsed.get("evidence", []))
    fields.extend(parsed.get("cautions", []))
    text = " ".join(str(value) for value in fields).lower()
    return [kw for kw in DEMOGRAPHIC_KEYWORDS if kw in text]


def parse_naming_response(response: str, expected_segment_id: int) -> tuple[dict | None, list[str]]:
    """LLM 원본 응답을 파싱·검증한다. 실패 시 (None, errors)를 반환한다."""
    parsed = _parse_json_block(response)
    errors = validate_naming_response(parsed, expected_segment_id)
    if errors:
        return None, errors

    warnings = check_demographic_language(parsed)
    if warnings:
        logger.warning(
            "[segment %s] demographic/lifestyle 표현 의심 단어 발견 (자동 거부 아님, 검토 필요): %s",
            expected_segment_id,
            warnings,
        )
    return parsed, []


DEFAULT_MODEL = "solar-pro"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_RETRIES = 2


def label_segment(
    client,
    row: pd.Series,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict:
    """세그먼트 하나에 대해 LLM naming을 호출·검증하고, 실패 시 재시도한다.

    temperature=0으로 고정해 재현성을 높인다(완전한 결정성은 보장되지 않음).
    """
    segment_id = int(row["segment_id"])
    user_prompt = build_user_prompt(row)
    total_attempts = max_retries + 1
    last_errors: list[str] = []

    for attempt in range(1, total_attempts + 1):
        try:
            response = call_llm(
                client, SEGMENT_NAMING_SYS, user_prompt, model=model, temperature=temperature
            )
        except Exception as e:
            last_errors = [f"LLM 호출 실패: {e}"]
            logger.error(
                "[segment %s] 시도 %d/%d 호출 실패: %s", segment_id, attempt, total_attempts, e
            )
            continue

        parsed, errors = parse_naming_response(response, segment_id)
        if parsed is not None:
            if attempt > 1:
                logger.info("[segment %s] %d번째 시도에서 성공", segment_id, attempt)
            return parsed

        last_errors = errors
        logger.warning(
            "[segment %s] 시도 %d/%d 검증 실패: %s", segment_id, attempt, total_attempts, errors
        )

    logger.error("[segment %s] 최대 재시도(%d회) 초과 — NAMING_FAILED", segment_id, total_attempts)
    return {"segment_id": segment_id, "status": "NAMING_FAILED", "errors": last_errors}


def merge_personas_with_customers(
    customer_segments: pd.DataFrame, personas: list[dict]
) -> pd.DataFrame:
    """segment_id 기준으로 naming 결과를 customer segment 테이블에 병합한다."""
    personas_df = pd.DataFrame(personas)
    for col in PERSONA_MERGE_COLUMNS:
        if col not in personas_df.columns:
            personas_df[col] = pd.NA

    for col in ["evidence", "cautions"]:
        personas_df[col] = personas_df[col].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
        )
    return customer_segments.merge(personas_df[PERSONA_MERGE_COLUMNS], on="segment_id", how="left")


def validate_merged_segments(
    merged: pd.DataFrame, customer_segments: pd.DataFrame, personas: list[dict]
) -> None:
    """병합 결과를 검증한다. 실패 시 ValueError를 발생시킨다."""
    errors = []

    if len(merged) != len(customer_segments):
        errors.append(f"병합 후 row 수 불일치: 원본={len(customer_segments)}, 병합후={len(merged)}")

    missing_mask = merged["segment_name"].isna()
    if missing_mask.any():
        missing_ids = sorted(merged.loc[missing_mask, "segment_id"].unique())
        errors.append(
            f"segment_name이 비어있는 row {missing_mask.sum()}개 (segment_id: {missing_ids})"
        )

    persona_segment_count = len({p["segment_id"] for p in personas})
    customer_segment_count = customer_segments["segment_id"].nunique()
    if persona_segment_count != customer_segment_count:
        errors.append(
            f"segment 수 불일치: personas={persona_segment_count}, "
            f"customer_segments={customer_segment_count}"
        )

    if errors:
        raise ValueError("세그먼트 병합 검증 실패:\n" + "\n".join(errors))


def next_run_label(date_str: str | None = None) -> str:
    """오늘 날짜 기준 다음 run 번호를 자동 부여한다 (run_YYYY-MM-DD_N)."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    prefix = f"run_{date_str}_"
    existing_nums = [
        int(p.name[len(prefix) :])
        for p in EXPERIMENTS_DIR.glob(f"{prefix}*")
        if p.is_dir() and p.name[len(prefix) :].isdigit()
    ]
    return f"{prefix}{max(existing_nums, default=0) + 1}"


def save_experiment_run(results: list[dict], run_label: str | None = None) -> Path:
    """naming 결과를 experiments/segment_naming_v2/run_X/segment_personas.json에 저장한다.

    canonical 파일(data/processed/segment_personas_v2*.json)은 건드리지 않는다 —
    채택 여부는 사람이 CHOICES.md에 기록한 뒤 --promote로 반영한다.
    """
    label = run_label or next_run_label()
    run_dir = EXPERIMENTS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "segment_personas.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("실험 저장: %s", output_path)
    return output_path


def promote_run(run_label: str) -> Path:
    """검토를 마친 run을 canonical 파일로 승격한다. API 호출 없음."""
    src = EXPERIMENTS_DIR / run_label / "segment_personas.json"
    if not src.exists():
        raise FileNotFoundError(f"run을 찾을 수 없음: {src}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, OUTPUT_PATH)
    logger.info("승격 완료: %s -> %s", src, OUTPUT_PATH)
    return OUTPUT_PATH


def merge_customer_segments() -> Path:
    """customer_segments_all_customers.csv와 segment_personas_v2.json을 병합해 저장한다.

    API 호출 없이 파일 병합만 수행한다 (Issue #26).
    """
    customer_segments = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        personas = json.load(f)

    merged = merge_personas_with_customers(customer_segments, personas)
    validate_merged_segments(merged, customer_segments, personas)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(LABELED_OUTPUT_PATH, index=False)
    logger.info("고객별 세그먼트 라벨 테이블 저장: %s", LABELED_OUTPUT_PATH)
    return LABELED_OUTPUT_PATH


def parse_args():
    parser = ArgumentParser(description="Segment summary 기반 LLM naming (Issue #17).")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="같은 설정으로 반복 실행할 실험 횟수. 매 실행은 별도 run 폴더에 저장된다.",
    )
    parser.add_argument(
        "--promote",
        metavar="RUN_LABEL",
        default=None,
        help=(
            "지정한 run(예: run_2026-07-03_3)의 결과를 canonical 파일로 승격한다. "
            "API 호출 없이 파일 복사만 수행한다."
        ),
    )
    parser.add_argument(
        "--merge-customers",
        action="store_true",
        help=(
            f"{CUSTOMER_SEGMENTS_PATH.name}와 {OUTPUT_PATH.name}을 segment_id 기준으로 병합해 "
            f"{LABELED_OUTPUT_PATH.name}을 생성한다 (Issue #26). API 호출 없음."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.promote:
        promote_run(args.promote)
        return

    if args.merge_customers:
        merge_customer_segments()
        return

    df = pd.read_csv(SEGMENT_SUMMARY_PATH).sort_values("segment_id")
    client = OpenAI(
        api_key=get_required_env("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1",
        timeout=60,
    )

    for i in range(1, args.n_runs + 1):
        results = [label_segment(client, row) for _, row in df.iterrows()]
        failed = [r for r in results if r.get("status") == "NAMING_FAILED"]
        logger.info(
            "naming 완료 (%d/%d): 성공 %d / 실패 %d",
            i,
            args.n_runs,
            len(results) - len(failed),
            len(failed),
        )
        save_experiment_run(results)


if __name__ == "__main__":
    main()
