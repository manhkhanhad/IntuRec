import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import random
from tqdm import tqdm
import multiprocessing
import glob
import os
import random
import json
import math

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    
class IntuRecF2Dataset(Dataset):
    def __init__(self, data_path, dataset, tokenizer, sample=-1, dataset_type="train", intuition_num=10, seed=0, reason_step=1):

        self.data_path = data_path
        self.dataset = dataset
        self.intuition_num = intuition_num
        self.reason_step = reason_step
        csv_files = glob.glob(os.path.join(f"{data_path}/{dataset}/{dataset_type}_extra", "*.json"))
        assert len(csv_files) == 1, "There should be only one CSV file in the directory."
        with open(csv_files[0], "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        info_files = glob.glob(os.path.join(f"{data_path}/{dataset}/info", "*.txt"))
        assert len(info_files) == 1, "There should be only one txt file in the directory."
        info_file = info_files[0]
        with open(info_file, "r", encoding="utf-8") as f:
            items = f.readlines()
            item_names = ["\"" + _.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') + "\"" for _ in items]
            self.item_names = item_names

        random.seed(seed)
        
        if sample > 0:
            rng = random.Random(seed)
            self.raw_data = rng.sample(self.raw_data, sample)
        self.tokenizer = Tokenizer(tokenizer)
        self.dataset_type = dataset_type
        self.category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
        category = self.category_dict[dataset]
        self.category = category
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.data = []
        self.raw_data_process()
        self.data_output = []
        self.text_data = []
        self.data_process()

    def __len__(self):
        return len(self.data)

    
    def raw_data_process(self):
        data = []
        for i in tqdm(range(len(self.raw_data))):
            row = self.raw_data[i]
            his_items = eval(row['history_item_title'])
            intuition_items = row[f'intuition_list_{self.intuition_num}']
            random_items = random.sample(self.item_names, self.intuition_num)
            tgt_item = "\"" + str(row['item_title']) + "\""
            tgt_ids = row["item_id"]
            data.append(
                {
                    "his_items": his_items,
                    "intuition_items": intuition_items,
                    "tgt_item": tgt_item,
                    "random_items": random_items,
                    "tgt_ids": tgt_ids
                }
            )
        self.data = data

    def data_process(self):
        reasoning_token = self.tokenizer.encode("<|Thought|>", bos=False, eos=False) 
        startofpredict_token = self.tokenizer.encode("<|StartPredict|>", bos=False, eos=False) 
        endofpredict_token = self.tokenizer.encode("<|EndPredict|>", bos=False, eos=False) 

        max_intuition_len = 0
        max_input_len = 0
        max_random_len = 0
        max_tgt_len = 0
        for i in range(len(self.data)):
            his_items = self.data[i]["his_items"]
            intuition_items = self.data[i]["intuition_items"]
            tgt_item = self.data[i]["tgt_item"]
            random_items = self.data[i]["random_items"]

            for j in range(len(intuition_items)):
                max_intuition_len = max(max_intuition_len, len(self.tokenizer.encode(intuition_items[j], bos=False, eos=False)))

            for j in range(len(random_items)):
                max_random_len = max(max_random_len, len(self.tokenizer.encode(random_items[j], bos=False, eos=False)))

            max_tgt_len = max(max_tgt_len, len(self.tokenizer.encode(tgt_item, bos=False, eos=False)))


            tgt_item_tokens = self.tokenizer.encode(tgt_item, bos=False, eos=False) 

            instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction: \n{self.instructs[0]}\n"""
            instruction_tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

            history_list = [f'\"{title}\"' for title in his_items]
            history = (
                f'### User Input: \n'
                f'The user has palyed the following {self.category}s before: '
                f'{", ".join(history_list)}.\n'
            )
            history_tokens = self.tokenizer.encode(history, bos=False, eos=False)

            answer_prefix = "### Response: \n"
            answer_prefix_tokens = self.tokenizer.encode(answer_prefix, bos=False, eos=False)

            token_part1 = instruction_tokens + history_tokens + answer_prefix_tokens + reasoning_token * self.reason_step + startofpredict_token
            token_part2 = tgt_item_tokens + endofpredict_token

            if self.dataset_type == "test":
                max_input_len = max(max_input_len, len(token_part1))
            else:
                max_input_len = max(max_input_len, len(token_part1 + token_part2))

        data_output = []
        text_data = []
        for i in tqdm(range(len(self.data))):
            his_items = self.data[i]["his_items"]
            intuition_items = self.data[i]["intuition_items"]
            tgt_item = self.data[i]["tgt_item"]
            random_items = self.data[i]["random_items"]
            tgt_ids = self.data[i]["tgt_ids"]

            tgt_item_tokens = self.tokenizer.encode(tgt_item, bos=False, eos=False) 

            instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n### Instruction: \n{self.instructs[0]}\n"""
            instruction_tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

            history_list = [f'\"{title}\"' for title in his_items]
            history = (
                f'### User Input: \n'
                f'The user has palyed the following {self.category}s before: '
                f'{", ".join(history_list)}.\n'
            )
            history_tokens = self.tokenizer.encode(history, bos=False, eos=False)

            answer_prefix = "### Response: \n"
            answer_prefix_tokens = self.tokenizer.encode(answer_prefix, bos=False, eos=False)

            token_part1 = instruction_tokens + history_tokens + answer_prefix_tokens + reasoning_token * self.reason_step + startofpredict_token
            token_part2 = tgt_item_tokens + endofpredict_token


            if self.dataset_type == "test":

                intuition_tokens_ = [self.tokenizer.encode(item, bos=False, eos=False) for item in intuition_items]
                intuition_tokens = [[self.tokenizer.tokenizer.pad_token_id]*(max_intuition_len-len(cur_intuition_tokens))+cur_intuition_tokens for cur_intuition_tokens in intuition_tokens_]

                attention_mask = [0] * (max_input_len - len(token_part1)) + [1] * len(token_part1)
                input_ids = [self.tokenizer.tokenizer.pad_token_id]*(max_input_len - len(token_part1)) + token_part1
                sample = {
                    "input_ids": np.array(input_ids),
                    "intuition_tokens": np.array(intuition_tokens),
                    "attention_mask": np.array(attention_mask),
                }
                text_sample = {
                    "instruction": instruction,
                    "history": history,
                    "intuition": intuition_items,
                    "answer_prefix": answer_prefix,
                    "all_input": instruction + history + answer_prefix + "<|Thought|>"*self.reason_step + "<|StartPredict|>"
                }
            else:

                intuition_tokens_ = [self.tokenizer.encode(item, bos=False, eos=False) for item in intuition_items]
                intuition_tokens = [[self.tokenizer.tokenizer.pad_token_id]*(max_intuition_len-len(cur_intuition_tokens))+cur_intuition_tokens for cur_intuition_tokens in intuition_tokens_]

                random_tokens_ = [self.tokenizer.encode(item, bos=False, eos=False) for item in random_items]
                random_tokens = [[self.tokenizer.tokenizer.pad_token_id]*(max_random_len-len(cur_random_tokens))+cur_random_tokens for cur_random_tokens in random_tokens_]

                tgt_tokens = [self.tokenizer.tokenizer.pad_token_id]*(max_tgt_len - len(tgt_item_tokens)) + tgt_item_tokens


                input_ids = token_part1 + token_part2
                attention_mask = [0] * (max_input_len - len(input_ids)) + [1] * len(input_ids)
                labels = [-100] * (max_input_len - len(input_ids)) + ([-100] * len(token_part1) + token_part2)
                input_ids = [self.tokenizer.tokenizer.pad_token_id]*(max_input_len - len(input_ids)) + input_ids

                sample = {
                    "input_ids": np.array(input_ids),
                    "attention_mask": np.array(attention_mask),
                    "intuition_tokens": np.array(intuition_tokens),
                    "random_tokens": np.array(random_tokens),
                    "tgt_tokens": np.array(tgt_tokens),
                    "tgt_ids": np.array(tgt_ids),
                    "labels": np.array(labels),
                }
                text_sample = {
                    "instruction": instruction,
                    "history": history,
                    "intuition": intuition_items,
                    "answer_prefix": answer_prefix,
                    "tgt_item": tgt_item,
                    "all_input": instruction + history + answer_prefix + "<|Thought|>"*self.reason_step + "<|StartPredict|>" + tgt_item + "<|EndPredict|>"
                }
            data_output.append(sample)
            text_data.append(text_sample)

        for sample in data_output:
            assert len(sample["input_ids"]) == max_input_len
            assert len(sample["attention_mask"]) == max_input_len
            for intuition_tokens_ in sample["intuition_tokens"]:
                assert len(intuition_tokens_) == max_intuition_len
            if "random_tokens" in sample:
                for random_tokens_ in sample["random_tokens"]:
                    assert len(random_tokens_) == max_random_len
            if "tgt_tokens" in sample:
                assert len(sample["tgt_tokens"]) == max_tgt_len
            if "labels" in sample:
                assert len(sample["labels"]) == max_input_len

        self.text_data = text_data
        self.data_output = data_output

    def __getitem__(self, idx):
        return self.data_output[idx]

