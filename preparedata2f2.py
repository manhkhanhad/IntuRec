import torch
import json
import os

beam_num = 15
dataset = "CDs_and_Vinyl" # CDs_and_Vinyl, Musical_Instruments, Toys_and_Games, Video_Games, Books
test_res = torch.load(f"data/{dataset}/test_beam_{beam_num}.pt")
valid_res = torch.load(f"data/{dataset}/valid_beam_{beam_num}.pt")
train_res = torch.load(f"data/{dataset}/train_beam_{beam_num+1}.pt")


with open(f"data/{dataset}/train_extra/extra.json") as f:
    train_data = json.load(f)
with open(f"data/{dataset}/valid_extra/extra.json") as f:
    valid_data = json.load(f)
with open(f"data/{dataset}/test_extra/extra.json") as f:
    test_data = json.load(f)

print(len(train_data))
print(len(train_res["outputs"]))
print(len(valid_data))
print(len(valid_res["outputs"]))
print(len(test_data))
print(len(test_res["outputs"]))

N = len(train_data) * 1.0
n = 0
for i in range(len(train_data)):
    tgt_i = "\"" + train_data[i]["item_title"] + "\""
    intuition_list = train_res["outputs"][i]
    if tgt_i in intuition_list:
        n += 1
train_hit = n/N
print(f"train_hit: {train_hit}")

N = len(valid_data) * 1.0
n = 0
for i in range(len(valid_data)):
    tgt_i = "\"" + valid_data[i]["item_title"] + "\""
    intuition_list = valid_res["outputs"][i]
    if tgt_i in intuition_list:
        n += 1
valid_hit = n/N
print(f"valid_hit: {valid_hit}")

N = len(test_data) * 1.0
n = 0
for i in range(len(test_data)):
    tgt_i = "\"" + test_data[i]["item_title"] + "\""
    intuition_list = test_res["outputs"][i]
    if tgt_i in intuition_list:
        n += 1
test_hit = n/N
print(f"test_hit: {test_hit}")

print(f"balance_ratio: {test_hit / train_hit}")