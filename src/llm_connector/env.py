"""Environment loading helpers for LLM scripts."""

import os

from dotenv import load_dotenv


def get_required_env(name: str) -> str:
    """Load .env and return a required environment variable."""
    load_dotenv()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다. .env 또는 셸 환경변수를 확인하세요.")
    return value
