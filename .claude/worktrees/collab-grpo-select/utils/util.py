import json
from tqdm import tqdm
import math
import numpy as np
def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

def get_prefix_data_inturec_f1(info_file, tokenizer):
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"" for _ in info]
    prefixID = [tokenizer(_).input_ids for _ in info]


    startofpredict_token = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
    endofpredict_token = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]

    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(endofpredict_token)
        for i in range(len(ID)):
            if i == 0:
                hash_number = get_hash([startofpredict_token])
            else:
                hash_number = get_hash(ID[:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    return hash_dict

def get_prefix_data_inturec_f2(info_file, tokenizer):
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"" for _ in info]
    prefixID = [tokenizer(_).input_ids for _ in info]


    startofpredict_token = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
    endofpredict_token = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]

    hash_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(endofpredict_token)
        for i in range(len(ID)):
            if i == 0:
                hash_number = get_hash([startofpredict_token])
            else:
                hash_number = get_hash(ID[:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    return hash_dict

def calRecallAndNdcgFromFile(result_path, item_path, predict_col_name="predict", ans_col_name="output"):

    error_item_num = 0

    with open(result_path, "r") as f:
        test_data = json.load(f)
        f.close()
    with open(item_path, "r") as f:
        items = f.readlines()
        item_names = [_.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') for _ in items]
        item_ids = [_ for _ in range(len(item_names))]
        item_dict = dict()
        for i in range(len(item_names)):
            if item_names[i] not in item_dict:
                item_dict[item_names[i]] = [item_ids[i]]
            else:   
                item_dict[item_names[i]].append(item_ids[i])
        
        topk_list = [1, 5, 10]

        ndcg_list = [0.0, 0.0, 0.0]
        recall_list = [0.0, 0.0, 0.0]


        text = [ [_.strip(" \n").strip("\"").strip(" ") for _ in sample[predict_col_name]] for sample in test_data]

        for index, sample in tqdm(enumerate(text)):

            if type(test_data[index][ans_col_name]) == list:
                target_item = test_data[index][ans_col_name][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index][ans_col_name].strip(" \n\"")
            hit_index = math.inf
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    error_item_num += 1
                if sample[i] == target_item:
                    hit_index = i
            for index, topk in enumerate(topk_list):
                if hit_index < topk:
                    ndcg_list[index] += (1 / math.log(hit_index + 2))
                    recall_list[index] += 1
        
        for i in range(len(topk_list)):
            ndcg_list[i] /= len(text) * (1.0 / math.log(2))
            recall_list[i] /= len(text)
        print(f"Res of {predict_col_name}")
        print(f"Recall  @1, 5, 10: {recall_list}")
        print(f"NDCG    @1, 5, 10: {ndcg_list}")
        print(f"Error Item Num: {error_item_num}")
        print(f"Total Item Num: {len(text)*10}")


def ndcg_at_k(pred, tgt, k):
    """
    Calculate NDCG at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for NDCG

    Returns:
    - ndcg: NDCG score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = top_k_preds == tgt[:, None]
    dcg_scores = relevant_mask.astype(np.float32) / np.log2(np.arange(2, k + 2))
    dcg = np.sum(dcg_scores, axis=1)
    idcg = 1.0
    ndcg = dcg / idcg
    return np.mean(ndcg)

def recall_at_k(pred, tgt, k):
    """
    Calculate Recall at K using NumPy.

    Args:
    - pred: Array of shape [B, K], predicted ranking for each user
    - tgt: Array of shape [B], ground truth relevant item for each user
    - k: int, rank position for recall

    Returns:
    - recall: Recall score at K for the batch
    """
    top_k_preds = pred[:, :k]
    relevant_mask = np.any(top_k_preds == tgt[:, None], axis=1)
    recall = relevant_mask.astype(np.float32)
    return np.mean(recall)


