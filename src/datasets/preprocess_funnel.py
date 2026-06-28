"""
퍼널 데이터셋을 MBA 포맷으로 변환하는 전처리 스크립트.

MBA 포맷 (세미콜론 구분):
  BillNo;Itemname;Quantity;Date;Price;CustomerID;Country

입력 (퍼널데이터셋 최종/):
  orders.csv, order_items.csv, products.csv, customers.csv

출력:
  ../data/funnel_mba_format.csv
"""

import pandas as pd

FUNNEL_DIR = "../../퍼널데이터셋 최종"
OUTPUT_PATH = "../../data/funnel_mba_format.csv"


def load_and_merge(funnel_dir: str) -> pd.DataFrame:
    orders = pd.read_csv(f"{funnel_dir}/orders.csv")
    order_items = pd.read_csv(f"{funnel_dir}/order_items.csv")
    products = pd.read_csv(f"{funnel_dir}/products.csv")

    # order_items + orders -> 고객 ID, 날짜, 국가
    df = order_items.merge(
        orders[["order_id", "customer_id", "order_time", "country"]], on="order_id", how="left"
    )

    # + products -> 상품명
    df = df.merge(products[["product_id", "name"]], on="product_id", how="left")

    return df


def to_mba_format(df: pd.DataFrame) -> pd.DataFrame:
    mba = pd.DataFrame(
        {
            "BillNo": df["order_id"].astype("int32"),
            "Itemname": df["name"].astype("string"),
            "Quantity": df["quantity"].astype("int32"),
            "Date": df["order_time"].astype("string"),
            "Price": df["unit_price_usd"].astype("string"),
            "CustomerID": df["customer_id"].astype("int32"),
            "Country": df["country"].astype("string"),
        }
    )

    before = len(mba)
    mba = mba.dropna()
    after = len(mba)
    if before != after:
        print(f"NaN 제거: {before - after}행 삭제")

    return mba


def main():
    print("데이터 로딩 및 조인 중...")
    df = load_and_merge(FUNNEL_DIR)

    print("MBA 포맷으로 변환 중...")
    mba = to_mba_format(df)

    unique_users = mba["CustomerID"].nunique()
    unique_items = mba["Itemname"].nunique()
    print(f"총 {len(mba):,}행 | 고객 {unique_users:,}명 | 상품 {unique_items:,}개")

    mba.to_csv(OUTPUT_PATH, sep=";", index=False)
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
