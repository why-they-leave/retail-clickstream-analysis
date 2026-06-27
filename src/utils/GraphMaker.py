import sys
sys.path.append('../')
from typing import Dict
import random_walker.Evaluation as eva
import numpy as np

def make_app(
    sampled_GT,     # {idx:[str]}
    persona_probs,  # np.array
    unlabeled_uidxs,# {uidx}
    app_scope_idxs,# {uidx}
    persona2idx_whole,  # {str: uidx}
    persona2idx_whole_2,# {str: uidx}
    persona_approach_num = 5,   #int
) -> Dict:

    multi_labels_approach = {}  # {uidx: ps}
    tri_graph_uidx2pidx_approach = {}   # remap from the
    
    multi_labels_approach.update(sampled_GT)
    persona_predictions = np.argsort(persona_probs, axis=1) # small to large
    unlabeled_uidxs_2_prediction = eva.make_unlabeled_uidxs_2_prediction(unlabeled_uidxs)
    whole_personas_2 = list(persona2idx_whole_2.keys())

    for uidx in app_scope_idxs:
        assert uidx not in multi_labels_approach
        pred_idx = unlabeled_uidxs_2_prediction[uidx] # position of uidx in prediction matrix
        persona_idxs_rank = persona_predictions[pred_idx][::-1] # rerank from large to small
        ps_approach_pidxs = persona_idxs_rank[:persona_approach_num] # select largest 5
        ps_approach = [whole_personas_2[pidx] for pidx in ps_approach_pidxs] # convert to the string
        multi_labels_approach[uidx] = ps_approach

    for uidx, ps in multi_labels_approach.items():
        # ignore the persona label of the unrepresentable users
        assert 'Unrepresentable' not in ps # already all representative
        ps_idx = [persona2idx_whole[p] for p in ps] # use the original persona2idx_whole to make the transformation
        tri_graph_uidx2pidx_approach[uidx] = ps_idx

    tri_graph_uidx2pidx_approach = {int(k):[int(e) for e in v] for k,v in tri_graph_uidx2pidx_approach.items()} # reform the datatype
    return tri_graph_uidx2pidx_approach