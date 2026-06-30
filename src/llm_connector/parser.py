"""
LLM 응답 JSON을 파싱해 페르소나 라벨을 추출한다.
"""

import ast
import json


def _parse_json_block(text: str) -> dict | None:
    """JSON 블록 추출 후 파싱. json → ast.literal_eval 순으로 시도."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    snippet = text[start:end]
    try:
        return json.loads(snippet)
    except Exception:
        try:
            return ast.literal_eval(snippet)  # [fix#6] eval() → ast.literal_eval() (임의 코드 실행 방지)
        except Exception:
            return None


def parse_user_response(uid, answer_str: str, defined_persona_set: set) -> dict | None:
    """LLM 응답에서 {uid: [persona_names]} 추출. 파싱 실패 시 None 반환."""
    uid_str = str(uid)
    res = _parse_json_block(answer_str)
    if res is None:
        return None

    val = (
        res.get(uid_str)
        or res.get(uid)
        or res.get(int(uid_str) if uid_str.isdigit() else uid)
    )
    if val is None:
        return None

    valid = [p for p in val if p in defined_persona_set or p == "Unrepresentable"]
    return {uid_str: valid} if valid else None


def parse_item_response(answer_str: str, itemname: str, defined_persona_set: set) -> list:
    """LLM 응답에서 상품에 해당하는 유효한 페르소나 목록 추출."""
    res = _parse_json_block(answer_str)
    if not res:
        return []
    val = list(res.values())[0]
    return [p for p in val if p in defined_persona_set]
