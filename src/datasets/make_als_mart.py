'''
# make_als_mart.py
ALS 학습 및 평가를 위한 전체 데이터마트를 생성하는 코드
학습 코드에서 timestemp에 맞게 split하여 train, test 진행
결과 csv 형식: [user_id, item_id, timestamp, total_score]
- timestamp는 가장 최근에 발생한 event의 시간(max)을 반영
'''
import pandas as pd

def generate_user_item_datamart():
    print("데이터 로딩 중...")
    # 1. 원본 데이터 불러오기
    events = pd.read_csv('data/raw/events.csv')
    sessions = pd.read_csv('data/raw/sessions.csv')
    
    print("데이터 전처리 시작...")
    # 2. timestamp 컬럼을 datetime 타입으로 안전하게 변환
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    
    # 3. 이벤트별 점수 가중치 딕셔너리 정의
    score_map = {
        'page_view': 1,
        'add_to_cart': 3,
        'checkout': 4,
        'purchase': 5
    }
    events['score'] = events['event_type'].map(score_map)
    
    # 4. 세션별로 장바구니(add_to_cart)에 담긴 상품 목록 추출
    # checkout, purchase 이벤트 시 product_id가 비어있으므로 해당 세션의 상품 리스트를 매핑
    cart_items = events[events['event_type'] == 'add_to_cart'][['session_id', 'product_id']].drop_duplicates()
    
    # 5. 상품 ID가 기본적으로 존재하는 이벤트와 존재하지 않는 이벤트 분리
    item_events = events[events['product_id'].notnull()].copy()
    session_events = events[events['product_id'].isnull()].copy()
    
    # 6. 상품 ID가 비어있는 checkout/purchase 이벤트에 해당 세션의 장바구니 상품들 매핑 (Row 확장)
    session_events = session_events.drop(columns=['product_id'])
    session_events_expanded = pd.merge(session_events, cart_items, on='session_id', how='inner')
    
    # 7. 분리했던 두 데이터를 다시 하나로 결합
    full_events = pd.concat([item_events, session_events_expanded], ignore_index=True)
    
    # 8. sessions 데이터와 결합하여 customer_id(user_id) 확보
    df_mart = pd.merge(full_events, sessions[['session_id', 'customer_id', 'country']], on='session_id', how='left')
    
    # 9. 컬럼명 가독성을 위해 변경 (customer_id -> user_id, product_id -> item_id)
    df_mart = df_mart.rename(columns={'customer_id': 'user_id', 'product_id': 'item_id'})
    
    print("1. 전체 유저 기준 데이터마트 생성 중...")
    # [전체 유저] df_mart에서 그룹화 진행
    datamart_all = df_mart.groupby(['user_id', 'item_id']).agg(
        timestamp=('timestamp', 'max'),
        total_score=('score', 'sum')
    ).reset_index()
    
    datamart_all['user_id'] = datamart_all['user_id'].astype(int)
    datamart_all['item_id'] = datamart_all['item_id'].astype(int)
    
    output_all = 'data/processed/als_datamart.csv'
    datamart_all.to_csv(output_all, index=False)
    print(f" -> 전체 데이터마트 완료 ({len(datamart_all):,} 행)")
    
    print("2. USA 유저 기준 데이터마트 생성 중...")
    # [수정 포인트] 집계가 완료된 datamart_all에는 country가 없으므로, 
    # 원본 세션 정보가 살아있는 df_mart에서 'US'인 행만 먼저 필터링한 후 groupby를 수행합니다.
    df_mart_usa = df_mart[df_mart['country'] == 'US'] 
    
    datamart_usa = df_mart_usa.groupby(['user_id', 'item_id']).agg(
        timestamp=('timestamp', 'max'),
        total_score=('score', 'sum')
    ).reset_index()
    
    datamart_usa['user_id'] = datamart_usa['user_id'].astype(int)
    datamart_usa['item_id'] = datamart_usa['item_id'].astype(int)
    
    output_usa = 'data/processed/als_datamart_us.csv'
    datamart_usa.to_csv(output_usa, index=False)
    print(f" -> USA 데이터마트 완료 ({len(datamart_usa):,} 행)")

if __name__ == "__main__":
    generate_user_item_datamart()