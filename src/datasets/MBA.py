import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


def MBA_load_data(ds_path='../data/interim/funnel_mba_format.csv', country=None, debug=False):
    # -> [mba_df, user_ids, user_num, user_ids_kv, item_names, item_num, items_kv, G_user, G_item]
    # 원본 구매 로그 CSV를 읽어서, 이후 알고리즘이 쓰기 쉬운 user-item bipartite graph로 바꾼다.
    """
    * user-item bipartite graph: 유저와 아이템을 노드로 놓고, "구매했다"를 선으로 연결한 그래프 -> 구매 기록을 그래프 형태로 바꾼 것 
    user끼리는 직접 연결 x, item끼리도 직접 연결 x, user와 item 사이만 연결 
    
    아래는 예시. 

      G_user = {
      0: {1, 2},   # user 0이 item 1, item 2를 샀다
      1: {2},      # user 1이 item 2를 샀다
      2: {3},      # user 2가 item 3을 샀다
  }

  반대 방향도 같이 저장합니다.

  G_item = {
      1: {0},      # item 1은 user 0이 샀다
      2: {0, 1},   # item 2는 user 0, user 1이 샀다
      3: {2},      # item 3은 user 2가 샀다
  }
    """

    print(f'Loading MBA dataset from path:{ds_path}')

    ## load dataset and basic clean
    mba_df = pd.read_csv(ds_path, sep=';')
    if debug: print(mba_df.head())

    # country 필터링
    if country is not None:
        mba_df = mba_df[mba_df['Country'] == country]
        print(f'country filter applied: {country} → {len(mba_df)} rows')
        if mba_df.empty:
            raise ValueError(f'country={country} 필터 결과가 비어 있습니다.')

    # 설명: CustomerID나 item명이 비어 있으면 제거한다.
    # clean nan rows
    if mba_df.isna().sum().sum() > 0:
        print('all nan eliminated')
        mba_df = mba_df.dropna()
    # transfer types
    mba_df['BillNo'] = mba_df['BillNo'].astype('int32')
    mba_df['Itemname'] = mba_df['Itemname'].astype('string')
    mba_df['Quantity'] = mba_df['Quantity'].astype('int32')
    mba_df['Date'] = mba_df['Date'].astype('string')
    mba_df['Price'] = mba_df['Price'].astype('string')
    mba_df['CustomerID'] = mba_df['CustomerID'].astype('int32')

    ## Identifications
    # for user nodes
    user_ids = mba_df['CustomerID'].unique() # CustomerID를 모아서 user_ids 생성
    user_num = len(user_ids)
    print(f'totally {user_num} unique users')
    user_ids.sort()
    user_ids_kv = {}
    for ui in range(user_num): # CustomerID를 뽑고 정렬한 다음 각 ID에 index를 붙인다
        user_ids_kv[user_ids[ui]] = ui

    # for item nodes
    item_names = mba_df['Itemname'].unique() # Itemname을 모아서 item_names 생성
    item_num = len(item_names)
    print(f'totally {item_num} unique items')
    # item_names.sort()
    items_kv = {}
    for ii in range(item_num): # ItemID를 뽑고 정렬한 다음에 각 ID에 index를 붙인다.
        items_kv[item_names[ii]] = ii

    ## construct the bi-partite graph
    G_user = {} # {uidx: [tidx,]}
    G_item = {} # {tidx: [uidx,]}

    for index,row in tqdm(mba_df.iterrows()):
        user_index = user_ids_kv[row['CustomerID']]
        item_index = items_kv[row['Itemname']]

        # update user side
        if G_user.get(user_index) is None:
            G_user[user_index] = {item_index}
        else:
            G_user[user_index].update([item_index])

        # update item side
        if G_item.get(item_index) is None:
            G_item[item_index] = {user_index}
        else:
            G_item[item_index].update([user_index])

    assert len(G_item.keys()) == item_num and len(G_user.keys()) == user_num

    return mba_df, user_ids, user_num, user_ids_kv, item_names, item_num, items_kv, G_user, G_item
