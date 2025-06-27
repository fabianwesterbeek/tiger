import numpy as np
from utils.intra_list_diversity import ILD
from utils.gini_coefficient import GiniCoefficient
import torch

def eva(pre, ground_truth, comi_ndcg=False, ild = False, ild_rep=None, calcGini=False):

    hit5, recall5, NDCG5, hit10, recall10, NDCG10, gini5, gini10 = (0, 0, 0, 0, 0, 0, 0, 0)
    Gini = GiniCoefficient()
    epsilon = 0.1 ** 10

    for i in range(len(ground_truth)):
        one_DCG5, one_recall5, IDCG5, one_hit5 = (0, 0, 0, 0)
        one_DCG10, one_recall10, IDCG10, one_hit10 = (0, 0, 0, 0)

        top_5_item = pre[i][0:5].tolist()
        top_10_item = pre[i][0:10].tolist()
        positive_item = ground_truth[i]

        for pos, iid in enumerate(top_5_item):
            if iid in positive_item:
                one_recall5 += 1
                one_DCG5 += 1 / np.log2(pos + 2)

        for pos, iid in enumerate(top_10_item):
            if iid in positive_item:
                one_recall10 += 1
                one_DCG10 += 1 / np.log2(pos + 2)

        if comi_ndcg:
            for pos in range(one_recall5):
                IDCG5 += 1 / np.log2(pos + 2)
            for pos in range(one_recall10):
                IDCG10 += 1 / np.log2(pos + 2)
        else:
            for pos in range(len(positive_item[:5])):
                IDCG5 += 1 / np.log2(pos + 2)
            for pos in range(len(positive_item[:10])):
                IDCG10 += 1 / np.log2(pos + 2)

        NDCG5 += one_DCG5 / max(IDCG5, epsilon)
        NDCG10 += one_DCG10 / max(IDCG10, epsilon)

        top_5_item = set(top_5_item)
        top_10_item = set(top_10_item)
        positive_item = set(positive_item)

        if len(top_5_item & positive_item) > 0:
            hit5 += 1
        if len(top_10_item & positive_item) > 0:
            hit10 += 1

        recall5 += len(top_5_item & positive_item) / max(len(positive_item), epsilon)
        recall10 += len(top_10_item & positive_item) / max(len(positive_item), epsilon)

        if calcGini:
            gini5 += Gini.calculate_list_gini(top_5_item)
            gini10 += Gini.calculate_list_gini(top_10_item)

    hit5 /= len(ground_truth)
    recall5 /= len(ground_truth)
    NDCG5 /= len(ground_truth)
    hit10 /= len(ground_truth)
    recall10 /= len(ground_truth)
    NDCG10 /= len(ground_truth)
    gini5 /= len(ground_truth)
    gini10 /= len(ground_truth)

    
    result = {
        'ndcg@5': round(NDCG5, 4), 'ndcg@10': round(NDCG10, 4),
        'hit@5': round(hit5, 4), 'hit@10': round(hit10, 4),
        'recall@5': round(recall5, 4), 'recall@10': round(recall10, 4),
        'gini@5': round(gini5, 4), 'gini@10': round(gini10, 4)
    }

    if ild:
        ild_calc = ILD()
        top_10_item = pre[i][0:10]
        #ild_rep = nn.Emdbeddiing
        top_10_item_tensor = torch.tensor(top_10_item, dtype=torch.long).to(ild_rep.emb_look_up.weight.device)
        top_10_item_emb = ild_rep(top_10_item_tensor)
        ild_10 = ild_calc.calculate_ild(top_10_item_emb, representation='st')
        result["ild@10"] = ild_10

    return result
