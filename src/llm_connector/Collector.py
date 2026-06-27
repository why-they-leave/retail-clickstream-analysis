from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

## user persona collections
def describe_user(uid, df, train_items=[]):
    # train_items: [item_name,]; only summarize for items in the list
    user_df = df[df['CustomerID'] == uid]
    items = dict(user_df[['Itemname', 'Quantity']].values)
    # further filter according to the given train_items variable
    if train_items:
        items = {k:v for k,v in items.items() if k in train_items}
    # add description like "36 PENCILS TUBE SKULLS purchased 16 times"
    items_description = '; '.join([f'{item}, {count} times' for item, count in items.items()])
    return f'The user {uid} has totally purchased {len(items)} unique products, we show each product name followed by its purchased times: he bought ' + items_description

def describe_users(uids, df):
    descriptions = []
    for uid in uids:
        descriptions.append(describe_user(uid, df))
    return '\n\n'.join(descriptions)

def assign_user_labels(
    client, # openai client object;
    grouped_item_df,    # pd.Dataframe;
    prompt_sys, prompt_user,    # str;
    uid,     # int;
    openai_model="gpt-4-0125-preview",   # openai model type
    debug=False, # bool;
) -> dict:

    transaction_data = describe_users([uid], grouped_item_df)
    
    prompt_user_tail = f"""Here is the data of user {uid}'s transaction data for you to analyze:{transaction_data}
    Remind one more time that you can only select from the given 20 personas' list and only use the exactly given persona, you cannot use other words to describe.
    You do not need to explain how you get the result, so please respond no more than the required format.
    """

    if debug:
        print(prompt_user+prompt_user_tail)
        return

    try:
        completion = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": prompt_sys},
                {"role": "user", "content": prompt_user+prompt_user_tail},
            ],
            stream=False,
        )

        response_result = ""
        # for chunk in stream:
        if completion.choices[0].message:
            response_result += completion.choices[0].message.content

        return {"user": uid, "users_profile": response_result}

    except Exception as e:  # Consider capturing a specific exception if possible
        print(f"[E] The following error occurred for user {uid} when collecting his persona labels: {e} ")
        return {"user": uid, "users_profile": "QUERY_FAILED"}

# define the parsing function
def from_json_to_obj(uid, answer_str, defined_persona_set, persona_fix_map):
    start_idx = 0
    end_idx = len(answer_str)
    
    # case 1: contains keyword 'json'
    json_index = answer_str.find('json')
    if json_index != -1:
        start_idx = json_index + 4
        end_idx = -3
    # case 2: no 'json' but start with '''
    elif answer_str.startswith("```"):
        start_idx = 3
        end_idx = -3
    
    res = eval(answer_str[start_idx:end_idx])
    
    assert type(res) == dict, 'oqweiuhd'
    assert res.get(uid) is not None, 'rtyuiold'

    # fix wrong persona names
    fixed_ps = []
    for p in res[uid]:
        if p in defined_persona_set:
            fixed_ps.append(p)
        else:
            fixed_ps.append(persona_fix_map[p])
    assert fixed_ps, 'empty'
    
    res = {uid: fixed_ps}
    
    return res


## item collections
def query_for_single_item(client, system_prompt, user_prompt, itemname):
    # itemname: str; e.g., "CHERRY BLOSSOM PURSE"
    try:
        completion = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + itemname},
            ],
            stream=False,
        )

        response_result = ""
        
        if completion.choices[0].message:
            response_result += completion.choices[0].message.content

        return {itemname: response_result}

    except Exception as e:  # Consider capturing a specific exception if possible
        print(f"Error occurred for item {itemname}: {e}")
        return {itemname: "QUERY_FAILED"}

def query_all_items(client, system_prompt, user_prompt, itemnames, workers=12):
    # itemnames: [itemname(str)]
    # -> results: [{itemname: response_text}]

    # collect procedure
    results = []
    print(f"Total items: {len(itemnames)}, start collecting...")
    
    with ThreadPoolExecutor(max_workers=workers) as executor: # avg. 0.13s per item
        
        future_results = [executor.submit(query_for_single_item, client, system_prompt, user_prompt, tname) for tname in itemnames]
    
        for future in tqdm(as_completed(future_results), total=len(itemnames), desc="Processing Items"):
            result = future.result()
            results.append(result)
    return results

def parse_item_response(predefined_persona_list, response):
    # response: {itemname(str): response_text(str)}
    # 1. try to parse directly
    itemname = list(response.keys())[0]
    response_text = response[itemname]
    # case1: "->'
    maohao_idx = response_text.find(':')
    if response_text[:maohao_idx].count('"') > 2:
        response_text = "{'" + response_text[2: maohao_idx-1] + "':" + response_text[maohao_idx+1:]
    try:
        adict = eval(response_text)
        itemname2 = list(adict.keys())[0]
        ps = adict[itemname2]
        ps2 = [p for p in ps if p in predefined_persona_list]
        return {itemname: ps2}
    except Exception as e:
        print(f'================== \n Error met when dealing {itemname}, details:')
        print(response_text)
        print(e)
        return {itemname: []}
