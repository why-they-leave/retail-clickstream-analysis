import numpy as np
from copy import deepcopy

def make_persona2idx_whole(persona2idx, predefined_persona_list):
    persona2idx_whole = {}
    persona2idx_whole.update(persona2idx)
    for p in [p for p in predefined_persona_list if p not in persona2idx]:
        persona2idx_whole[p] = len(persona2idx_whole)
    assert len(persona2idx_whole) == len(predefined_persona_list), 'poqtsohs' # check
    return persona2idx_whole

def make_unlabeled_uidxs_2_prediction(unlabeled_uidxs):
    return {uidx:idx for idx,uidx in enumerate(unlabeled_uidxs)} # align index between uidx and prediction idx

def f1score(correct, pred):
    tp,fn,fp = 0,0,0
    correct_set = set(correct)
    pred_set = set(pred)
    tp = len([e for e in correct_set if e in pred_set])
    fn = len(correct) - tp
    fp = len(pred) - tp
    f1 = (2 * tp)/(2 * tp + fp + fn)
    return f1

def recall(correct, pred):
    tp,fn,fp = 0,0,0
    correct_set = set(correct)
    pred_set = set(pred)
    tp = len([e for e in correct_set if e in pred_set])
    fn = len(correct) - tp
    fp = len(pred) - tp
    recall = tp / (tp + fn)
    return recall

def persona_prediction_eval(
    persona_probs,  # rank_res_labeled @ label_matrix_tuned
    label_matrix,
    test_scope_idxs,    # [uidx,]; test uidxs
    unlabeled_uidxs,    # [uidx,]; all unlabeled uidxs
    oracle, # Persona_Oracle object, for ground truth
    predefined_persona_list,    # [str,]; list of predefined personas
    persona2idx,    # {persona name: pidx};
    persona_list,
    sampled_GT, # sampled ground truths
    compare_len_para = 0,    # compare length, set to 0 means same as the gt_len
    f1score_k = 5,
    recall_k = 5,
    trivial = False,
    randomly = False,
):
    ## Accuracy Evaluation

    # 1. Generate predictions and statistics
    predictions = np.argsort(persona_probs, axis=1) # small to large

    # get gt for the test scope
    gt = oracle.query(test_scope_idxs)

    # preparations
    sta_s = {}
    sta_p = {p:[] for p in np.arange(len(predefined_persona_list))}
    sta_p_f1 = {p:{'tp':0, 'tn':0, 'fp':0, 'fn':0,} for p in np.arange(len(predefined_persona_list))}
    f1scores = {}
    all_f1 = []
    all_recall = []

    # unlabeled_uidxs_2_prediction = {uidx:idx for idx,uidx in enumerate(unlabeled_uidxs)} # align index between uidx and prediction idx
    unlabeled_uidxs_2_prediction = make_unlabeled_uidxs_2_prediction(unlabeled_uidxs)

    persona2idx_whole = make_persona2idx_whole(persona2idx, predefined_persona_list)

    # trivial baselines
    if trivial:
        persona_sta = {}
        for ps in sampled_GT.values():
            for p in ps:
                if persona_sta.get(p) is None: persona_sta[p] = 0
                persona_sta[p] += 1
        personas_trivial = sorted(persona_sta, key=lambda k:persona_sta[k], reverse=True)
        pred_personas_trivial = [persona2idx_whole[p] for p in personas_trivial]

    # start evaluation for each test gt
    for uidx, ps in gt.items():
        gt_personas = [persona2idx_whole[p] for p in ps] # ground truth personas, may include external labels
        
        assert gt_personas, user_ids[uidx] # gt_personas should be non-empty
        
        pidx = unlabeled_uidxs_2_prediction[uidx] # prediction matrix index
        pred_personas = predictions[pidx][::-1] # reorder from the largest membership to the least
        if trivial:
            pred_personas = pred_personas_trivial
        if randomly:
            pred_personas = deepcopy(pred_personas)
            np.random.shuffle(pred_personas)

        # start to compare
        gt_len = len(gt_personas)

        if compare_len_para <= 0:
            compare_len = gt_len
        else:
            compare_len = min(gt_len, compare_len_para)
        tmp = 0 # correctly answered count

        # s-statistics
        for i in range(compare_len):
            if pred_personas[i] in gt_personas: # gt_personas has no order, pred_personas has
                tmp += 1 # the overlap size
        
        correct_rate = tmp / compare_len
        
        # for persona-lenth sta
        if sta_s.get(gt_len) is None:
            sta_s[gt_len] = []
        sta_s[gt_len].append(correct_rate)

        if f1score_k > 0:
            if f1scores.get(gt_len) is None: f1scores[gt_len] = []
            f1s = f1score(gt_personas, pred_personas[:f1score_k])
            f1scores[gt_len].append(f1s)
            all_f1.append(f1s)
        
        if recall_k > 0:
            rcl = recall(gt_personas, pred_personas[:recall_k])
            all_recall.append(rcl)
        
        # p-statistics: for different persona classes
        for p in gt_personas:
            if p in pred_personas[:compare_len]:
                sta_p[p].append(1)
            else:
                sta_p[p].append(0)

        for p in sta_p_f1.keys():
            if p in pred_personas[:max(compare_len_para, compare_len)]:
                if p in gt_personas: # tp
                    sta_p_f1[p]['tp'] += 1
                else: # fp
                    sta_p_f1[p]['fp'] += 1
            else: # not in pred
                if p not in gt_personas: # tn
                    sta_p_f1[p]['tn'] += 1
                else: # fn
                    sta_p_f1[p]['fn'] += 1

    # return persona2idx_whole
    # 2. Process the sta and demo
    # s
    # consider sampled gt
    sta_labeled = {}
    for uid,ps in sampled_GT.items():
        p_num = len(ps)
        if sta_labeled.get(p_num) is None:
            sta_labeled[p_num] = 0
        sta_labeled[p_num] += 1

    # other predicted
    # print(sta_s)
    all_values = []
    for li in list(sta_s.values()):
        all_values += li
    assert len(all_values) == len(test_scope_idxs)
    overall_acc = np.mean(all_values)

    sta_s_avg = {}
    for gt_len in sta_s.keys():
        sta_s_avg[gt_len] = [np.mean(sta_s[gt_len]), len(sta_s[gt_len])] # problematic
    
    f1scores_avg = {}
    for gt_len in f1scores.keys():
        f1scores_avg[gt_len] = [np.mean(f1scores[gt_len]), len(f1scores[gt_len])]

    whole_acc = (len(all_values)*overall_acc + len(sampled_GT)*1.0)/(len(all_values)+len(sampled_GT))
    print(f'Overall Acc: {overall_acc :f} | {whole_acc :f}')
    print(f'Overall F1Score@{f1score_k}: {np.mean(all_f1)}')
    print(f'Overall Recall@{recall_k}: {np.mean(all_recall)}')

    res1 = []
    res3 = [] # f1
    for gt_l in sorted(list(sta_s_avg.keys())):
        acc = sta_s_avg[gt_l][0]
        count = sta_s_avg[gt_l][1]
        count2 = sta_labeled.get(gt_l, 0)
        whole_acc = (acc*count + 1.0*count2)/(count+count2)
        print(f'{gt_l}-persona Acc: {acc :f} (total {count}) | {whole_acc :f} (total {count+count2})')
        res1.append((acc, whole_acc))
        
        # if f1score_k > 0:
        #     print(f'{gt_l}-persona F1: {f1scores[gt_l][0] :f} (total {f1scores[gt_l][1]})')
        #     res3.append((acc, whole_acc))

    # p
    res2 = []
    perper_measure = {p:[np.mean(sta) if sta else 0.0, len(sta)] for p,sta in sta_p.items()}
    perper_sum = np.sum(label_matrix, axis=0) # count the number of labeled user for each persona class
    for i in range(len(persona2idx)):
        occupy = perper_sum[i]
        count = perper_measure[i][1]
        acc = perper_measure[i][0]
        whole_acc = (occupy*1.0 + count*acc)/(occupy + count)
        print(f'Persona {i} occupies {occupy} with Acc {acc :f} \t (total {count}) \t {whole_acc :f} \t (total {count+occupy})\t {persona_list[i]}')
        res2.append((acc, whole_acc))

    # print(perper_measure)
    return (overall_acc, whole_acc), res1, res2, sta_p_f1
