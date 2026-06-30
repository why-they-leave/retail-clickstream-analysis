"""
유저 데이터를 LLM 프롬프트용 텍스트로 변환한다.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def describe_user(
    uid, df, train_items=None, customer_features=None
) -> str:  # [fix#2] mutable default → None
    """유저 구매이력과 행동 컨텍스트를 LLM 프롬프트용 텍스트로 변환."""
    user_df = df[df["CustomerID"] == uid]
    items = dict(user_df[["Itemname", "Quantity"]].values)
    if train_items:
        items = {k: v for k, v in items.items() if k in train_items}

    items_desc = "; ".join([f"{item}, {count} times" for item, count in items.items()])
    text = (
        f"The user {uid} has totally purchased {len(items)} unique products. "
        f"Each product name is followed by its purchased times: {items_desc}."
    )

    if customer_features is not None:
        # [fix#3] customer_id 컬럼 없으면 경고 후 행동 컨텍스트 생략
        if (
            not isinstance(customer_features, pd.DataFrame)
            or "customer_id" not in customer_features.columns
        ):
            logger.warning("customer_features에 customer_id 컬럼이 없음 — 행동 컨텍스트 생략")
            return text

        row = customer_features[customer_features["customer_id"] == uid]
        if not row.empty:
            r = row.iloc[0]

            # [fix#4] or 0 패턴은 NaN을 걸러내지 못함 → pd.notna() 기반으로 교체
            pv = r["page_view_count"] if pd.notna(r.get("page_view_count")) else 0
            atc = r["add_to_cart_count"] if pd.notna(r.get("add_to_cart_count")) else 0
            atc_pv = f"{atc / pv * 100:.1f}%" if pv > 0 else "N/A"

            last_session = (
                f"{int(r['recency_session_days'])} days ago"
                if pd.notna(r.get("recency_session_days"))
                else "N/A"
            )
            last_order = (
                f"{int(r['recency_order_days'])} days ago"
                if pd.notna(r.get("recency_order_days"))
                else "N/A"
            )

            total_spend = f"${r['total_spend']:.2f}" if pd.notna(r.get("total_spend")) else "N/A"
            avg_order = (
                f"${r['avg_order_value']:.2f}" if pd.notna(r.get("avg_order_value")) else "N/A"
            )

            top_view = r.get("top_view_category") if pd.notna(r.get("top_view_category")) else "N/A"
            top_purchase = (
                r.get("top_purchase_category")
                if pd.notna(r.get("top_purchase_category"))
                else "N/A"
            )

            session_count = r["session_count"] if pd.notna(r.get("session_count")) else 0  # [fix#4]
            order_count = r["order_count"] if pd.notna(r.get("order_count")) else 0  # [fix#4]

            text += (
                f"\n\nBehavioral context:"
                f"\n- Sessions: {int(session_count)}"
                f"\n- Page views: {int(pv)}, Add-to-carts: {int(atc)} (ATC/PV: {atc_pv})"
                f"\n- Orders: {int(order_count)}, Total spend: {total_spend},"
                f" Average order value: {avg_order}"
                f"\n- Last session: {last_session}, Last order: {last_order}"
                f"\n- Most viewed category: {top_view}, Most purchased category: {top_purchase}"
            )

    return text


def describe_users(uids, df) -> str:
    """여러 유저의 구매이력을 하나의 텍스트로 변환."""
    return "\n\n".join([describe_user(uid, df) for uid in uids])
