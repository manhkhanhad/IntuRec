from transformers import (
    AutoTokenizer, 
    GenerationConfig,
    LogitsProcessorList, 
    TemperatureLogitsWarper
)
from model.IntuRecF1 import IntuRecF1
import torch
import glob
import pandas as pd
from typing import List
from data.IntuRecF1Dataset import IntuRecF1Dataset
from utils.util import get_hash, get_prefix_data_inturec_f1
import os
from tqdm import tqdm
from model.LogitsProcessor import IntuRecLogitsProcessor
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", type=str, default="train", help="dataset type")
    parser.add_argument("--beam_num", type=int, default=10, help="beam number")
    parser.add_argument("--dataset", type=str, default="CDs_and_Vinyl", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="length penalty")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--slice_num", type=int, default=-1)
    parser.add_argument("--slice_no", type=int, default=0)
    parser.add_argument("--tail_fix", action="store_true")

    return parser.parse_args()

def get_device(args):
    return torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")


args = parser_args()
device = get_device(args)

# Please fill in according to your configuration.
f1_ckpt_path = "your inturec f1 checkpoint path"

tokenizer = AutoTokenizer.from_pretrained(f1_ckpt_path, trust_remote_code=True) 

model = IntuRecF1.from_pretrained(
    f1_ckpt_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False
model.to(device)
model.eval()



data = IntuRecF1Dataset(
    "./data", 
    args.dataset, 
    tokenizer, 
    sample=-1,
    dataset_type=args.dataset_type,
    seed=0
)
data.dataset_type = "test"
data.data_process()


raw_data = [data[i] for i in range(len(data))]
block_num = (len(raw_data) + args.batch_size - 1) // args.batch_size
batch_raw_data = [raw_data[i * args.batch_size: (i + 1) * args.batch_size] for i in range(block_num)]

if args.slice_num != -1:
    per_slice_size = block_num // args.slice_num
    if args.tail_fix:
        batch_raw_data = batch_raw_data[args.slice_num * per_slice_size:]
    else:
        batch_raw_data = batch_raw_data[args.slice_no * per_slice_size: (args.slice_no + 1) * per_slice_size]


info_files = glob.glob(os.path.join(f"./data/{args.dataset}/info", f"{args.dataset}*.txt"))
assert len(info_files) == 1, "There should be only one txt file in the directory."
info_file = info_files[0]

hash_dict = get_prefix_data_inturec_f1(info_file, tokenizer)

def prefix_allowed_tokens_fn(batch_id, input_ids):
    hash_number = get_hash(input_ids)
    if hash_number in hash_dict:
        return hash_dict[hash_number]
    return []

generation_config = GenerationConfig(
    num_beams=args.beam_num,
    length_penalty=args.length_penalty,
    num_return_sequences=args.beam_num,
    pad_token_id = model.config.pad_token_id,
    eos_token_id = model.config.eos_token_id,
    max_new_tokens = 128,
)
with torch.no_grad():
    outputs = []
    scores = []
    for idx, data_i in enumerate(tqdm(batch_raw_data)):
        maxLen = max([len(_["input_ids"]) for _ in data_i])

        padding_data = {"input_ids": [], "attention_mask": []}

        for  _ in data_i:
            L = len(_["input_ids"])
            padding_data["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            padding_data["attention_mask"].append([0] * (maxLen - L) + _["attention_mask"])


        logitsprocessor1 = TemperatureLogitsWarper(temperature=args.temperature)
        logitsprocessor2 = IntuRecLogitsProcessor(
            tokenizer=tokenizer,
            model=model,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=args.beam_num,
            guide_token="<|StartPredict|>",
        )
        logitsprocessor = LogitsProcessorList([logitsprocessor1, logitsprocessor2])
        generation_output = model.generate(
            torch.tensor(padding_data["input_ids"]).to(model.device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logitsprocessor,
            attention_mask=torch.tensor(padding_data["attention_mask"]).to(model.device),
        )
        
        startofpredict_id = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
        endofpredict_id = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]

        generation_sequences = generation_output.sequences.detach().cpu()  # (bsz*beam, seq_len)
        mask_s1 = generation_sequences.eq(startofpredict_id)        # bool (bsz*beam, seq)
        assert mask_s1.sum(dim=1).eq(1).all()
        pos_s1 = mask_s1.long().argmax(dim=1)              # (bsz)

        mask_s2 = generation_sequences.eq(endofpredict_id)        # bool (bsz*beam, seq)
        assert mask_s2.sum(dim=1).eq(1).all()
        pos_s2 = mask_s2.long().argmax(dim=1)              # (bsz)

        arange_ids = torch.arange(mask_s1.shape[-1], device=generation_sequences.device).unsqueeze(0)  # (1, seq_len)
        mask_between = (arange_ids > pos_s1.unsqueeze(1)) & (arange_ids < pos_s2.unsqueeze(1))  # (bsz*beam, seq_len)
        segments = generation_sequences.masked_select(mask_between)  # (sum of segment lengths,)
        lengths = (mask_between.sum(dim=1)).tolist()
        segments_split = torch.split(segments, lengths)

        generation_res = tokenizer.batch_decode(segments_split, skip_special_tokens=True)
        generation_scores = generation_output.sequences_scores.detach().cpu().tolist()

        outputs_i = [generation_res[i * args.beam_num: (i + 1) * args.beam_num] for i in range(len(generation_res) // args.beam_num)]
        scores_i = [generation_scores[i * args.beam_num: (i + 1) * args.beam_num] for i in range(len(generation_scores) // args.beam_num)]

        outputs.extend(outputs_i)
        scores.extend(scores_i)

        del generation_output

        torch.cuda.empty_cache()

save_data = {
    "outputs": outputs,
    "scores": scores
}
if args.slice_num != -1:
    if args.tail_fix:
        torch.save(save_data, f"./data/{args.dataset}/{args.dataset_type}_beam_{args.beam_num}_slice_{args.slice_num}.pt")
    else:
        torch.save(save_data, f"./data/{args.dataset}/{args.dataset_type}_beam_{args.beam_num}_slice_{args.slice_no}.pt")
else:
    torch.save(save_data, f"./data/{args.dataset}/{args.dataset_type}_beam_{args.beam_num}.pt")
