"""
하위 호환성 shim — 기존 코드에서 Collector를 import하는 경우를 위해 유지한다.
새 코드는 formatter / client / parser를 직접 import한다.
"""

from llm_connector.client import call_llm, run_parallel
from llm_connector.formatter import describe_user, describe_users
from llm_connector.parser import parse_item_response, parse_user_response

__all__ = [
    "describe_user",
    "describe_users",
    "call_llm",
    "run_parallel",
    "parse_user_response",
    "parse_item_response",
]
