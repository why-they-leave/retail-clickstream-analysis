"""페르소나 파이프라인 설정 로딩 유틸리티."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/persona/params.yaml"


DEFAULT_CONFIG = {
    "sample": {
        "ratio": 0.025,
        "random_state": 42,
        "country": None,
    },
    "paths": {
        "input": "data/interim/funnel_mba_format.csv",
        "output_dir": "data/interim/funnel_persona_gen",
        "customer_features": "data/processed/customer_features_all_customers.csv",
    },
    "llm": {
        "model": "solar-pro",
        "max_workers": 12,
    },
    "persona": {
        "n_iterations": 40,
        "users_per_iteration": 100,
        "step2_iterations": 8,
        "sets_per_step2_sample": 5,
        "train_ratio": 0.8,
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_persona_config(config_path: str | Path | None = None) -> dict:
    """Load persona pipeline config with conservative defaults."""
    path = resolve_path(config_path) if config_path else DEFAULT_CONFIG_PATH
    loaded = {}
    if config_path and not path.exists():
        raise FileNotFoundError(f"config 파일을 찾을 수 없음: {path}")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULT_CONFIG, loaded)
    config["paths"]["input"] = resolve_path(config["paths"]["input"])
    config["paths"]["output_dir"] = resolve_path(config["paths"]["output_dir"])
    config["paths"]["customer_features"] = resolve_path(config["paths"]["customer_features"])
    config["paths"]["persona"] = resolve_path(
        config["paths"].get("persona", config["paths"]["output_dir"] / "final_personas.json")
    )
    return config
