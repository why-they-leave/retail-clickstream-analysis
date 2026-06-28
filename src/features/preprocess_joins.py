"""
raw 테이블을 조인해 중간 테이블을 생성한다.

출력:
    data/interim/session_events.csv   — sessions + events + products
    data/interim/order_details.csv    — orders + order_items + products
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("../../data/raw")
INTERIM_DIR = Path("../../data/interim")


def build_session_events(raw_dir: Path) -> pd.DataFrame:
    """sessions → events → products 조인."""
    sessions    = pd.read_csv(raw_dir / "sessions.csv")
    events      = pd.read_csv(raw_dir / "events.csv")
    products    = pd.read_csv(raw_dir / "products.csv")

    df = (
        sessions[["session_id", "customer_id", "start_time"]]
        .merge(
            events[["session_id", "event_type", "product_id"]],
            on="session_id",
            how="inner",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
        )
    )
    return df


def build_order_details(raw_dir: Path) -> pd.DataFrame:
    """orders → order_items → products 조인."""
    orders      = pd.read_csv(raw_dir / "orders.csv")
    order_items = pd.read_csv(raw_dir / "order_items.csv")
    products    = pd.read_csv(raw_dir / "products.csv")

    df = (
        orders[["order_id", "customer_id", "order_time", "total_usd"]]
        .merge(
            order_items[["order_id", "product_id", "quantity"]],
            on="order_id",
            how="left",
        )
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
        )
    )
    return df


def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    print("sessions_events_products 생성 중...")
    session_events = build_session_events(RAW_DIR)
    session_events.to_csv(INTERIM_DIR / "sessions_events_products.csv", index=False)
    print(f"  → {len(session_events):,}행 저장: {INTERIM_DIR / 'sessions_events_products.csv'}")

    print("orders_items_products 생성 중...")
    order_details = build_order_details(RAW_DIR)
    order_details.to_csv(INTERIM_DIR / "orders_items_products.csv", index=False)
    print(f"  → {len(order_details):,}행 저장: {INTERIM_DIR / 'orders_items_products.csv'}")


if __name__ == "__main__":
    main()
