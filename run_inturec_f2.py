import argparse
import math
import os
import random
import numpy as np
from transformers import (
    AutoTokenizer, 
    GenerationConfig,
    LogitsProcessorList, 
    TemperatureLogitsWarper
)
import torch
from data.IntuRecF2Dataset import IntuRecF2Dataset
from model.IntuRecF2 import IntuRecF2
import glob
from tqdm import tqdm
import json
from utils.util import get_hash, get_prefix_data_inturec_f2, calRecallAndNdcgFromFile
from model.LogitsProcessor import IntuRecLogitsProcessor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import datetime
MODEL_NAME="IntuRecF2"

def parser_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", choices=["CDs_and_Vinyl", "Toys_and_Games", "Video_Games", "Musical_Instruments", "Books"], default="CDs_and_Vinyl")
    parser.add_argument("--sample", type=int, default=-1)

    # Please fill in according to your configuration.
    parser.add_argument("--base_model", type=str, default="your llm model path")
    parser.add_argument("--intuition_num", type=int, default=10)

    parser.add_argument("--nums_epoch", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=120)
    parser.add_argument("--train_micro_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--cutoff_len", type=int, default=512)
    # Please fill in according to your configuration.
    parser.add_argument("--output_dir", type=str, default=f"your output ckpt path")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--reason_step", type=int, default=1)


    parser.add_argument("--train_or_test", type=str, default="train")  # train / test
    parser.add_argument("--decode_method", type=str, default="standard")  # standard / d3
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--data_result_root_path", type=str, default="result_data")
    # Optional exact folders (<root>/<dataset>/<experiment>/<run_name>); unset = original paths.
    parser.add_argument("--ckpt_dir", type=str, default="", help="Exact checkpoint folder (overrides output_dir layout).")
    parser.add_argument("--result_dir", type=str, default="", help="Exact result folder (overrides data_result_root_path layout).")

    return parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(args):
    return torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

def initial_model(args, device):
    if args.train_or_test == "test":
        args.base_model = args.ckpt_dir or f"{args.output_dir}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}/intuition_num_{args.intuition_num}/alpha_{args.alpha}/standard/rs_{args.reason_step}"
    else:
        args.base_model = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True) 
    if args.train_or_test == "train":
        additional_special_tokens = tokenizer.additional_special_tokens
        additional_special_tokens.append("<|Thought|>")
        additional_special_tokens.append("<|StartPredict|>")
        additional_special_tokens.append("<|EndPredict|>")
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        tokenizer.padding_side = "left"

    model = IntuRecF2.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
    ).to(device)
    if args.train_or_test == "train":
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EndPredict|>")
        model.config.use_cache = False
    model.alpha = args.alpha
    model.reason_step = args.reason_step

    return model, tokenizer

def initial_optimizer_scheduler(
    model, 
    lr, 
    wd, 
    optimizer, 
    epochs,  
    batchnum_per_epoch, 
    scheduler_type,
    warmup_ratio=None,
    warmup_step=None
):
    # optimizer 
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )
    else:
        raise ValueError("Invalid optimizer.")
    
    # scheduler
    total_steps = epochs * batchnum_per_epoch
    if warmup_ratio is not None:
        warmup_steps = int(total_steps * warmup_ratio)
    elif warmup_step is not None:
        warmup_steps = warmup_step
    else:
        warmup_steps = 0

    def lr_lambda_cos(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    def lr_lambda_linear(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_cos if scheduler_type == "cosine" else lr_lambda_linear
    ) if scheduler_type != "none" else None

    return optimizer, scheduler

def initial_dataLoader(args, tokenizer):

    datasets = {
        "train": IntuRecF2Dataset(
            args.data_path, 
            args.dataset, 
            tokenizer, 
            sample=args.sample,
            dataset_type="train",
            intuition_num=args.intuition_num,
            reason_step=args.reason_step,
            seed=args.seed
        ),
        "valid": IntuRecF2Dataset(
            args.data_path, 
            args.dataset, 
            tokenizer, 
            sample=args.sample,
            dataset_type="valid",
            intuition_num=args.intuition_num,
            reason_step=args.reason_step,
            seed=args.seed
        ),
        "test": IntuRecF2Dataset(
            args.data_path, 
            args.dataset, 
            tokenizer, 
            sample=args.sample,
            dataset_type="test",
            intuition_num=args.intuition_num,
            reason_step=args.reason_step,
            seed=args.seed
        )
    }
    print(f"Train dataset sample: \n{datasets['train'].text_data[0]['all_input']}\n")
    print(f"Valid dataset sample: \n{datasets['valid'].text_data[0]['all_input']}\n")
    print(f"Test dataset sample: \n{datasets['test'].text_data[0]['all_input']}\n")

    dataloaders = {
        "train": DataLoader(
            datasets["train"], 
            batch_size=args.train_micro_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        ),
        "valid": DataLoader(
            datasets["valid"], 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"], 
            batch_size=args.eval_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    }

    return dataloaders["train"], dataloaders["valid"], dataloaders["test"]


def train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, args):
    output_dir = args.ckpt_dir or f"{args.output_dir}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}/intuition_num_{args.intuition_num}/alpha_{args.alpha}/rs_{args.reason_step}"
    os.makedirs(output_dir, exist_ok=True)

    min_valid_loss = float("inf")

    gradient_accumulation_steps = args.train_batch_size // args.train_micro_batch_size
    for epoch in range(args.nums_epoch):
        print(f"Epoch [{epoch + 1}] - Start Training")
        model.train()
        total_loss = []
        total_ntp_loss = []
        total_intuition_loss = []
        step_loss = 0
        step_ntp_loss = 0
        step_intuition_loss = 0
        optimizer.zero_grad()
        time_start = time.time()
        for ministep, minibatch in enumerate(train_loader):
            minibatch = {k: v.to(device) for k, v in minibatch.items()}
            input_ids = minibatch["input_ids"]
            attention_mask = minibatch["attention_mask"]
            intuition_tokens = minibatch["intuition_tokens"]
            random_tokens = minibatch["random_tokens"]
            tgt_tokens = minibatch["tgt_tokens"]
            tgt_ids = minibatch["tgt_ids"]
            labels = minibatch["labels"]

            res = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intuition_tokens=intuition_tokens,
                random_tokens=random_tokens,
                tgt_tokens=tgt_tokens,
                tgt_ids=tgt_ids,
                labels=labels,
            )
            loss = res.loss
            ntp_loss = res.ntp_loss
            intuition_loss = res.intuition_loss

            step_loss += loss.item()
            step_ntp_loss += ntp_loss.item()
            step_intuition_loss += intuition_loss.item()
            (loss / gradient_accumulation_steps).backward()

            if (ministep + 1) % gradient_accumulation_steps == 0:
                step_loss = step_loss / gradient_accumulation_steps
                step_ntp_loss = step_ntp_loss / gradient_accumulation_steps
                step_intuition_loss = step_intuition_loss / gradient_accumulation_steps
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss.append(step_loss)
                total_ntp_loss.append(step_ntp_loss)
                total_intuition_loss.append(step_intuition_loss)

                if (ministep + 1) % (10 * gradient_accumulation_steps) == 0:
                    time_end = time.time()
                    duration = time_end - time_start
                    time_start = time_end
                    if scheduler is not None:
                        print(
                            f"Step [{(ministep + 1) // gradient_accumulation_steps}/{(len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps}] - "
                            f"Avg Total Loss: {np.mean(total_loss):.4f}, "
                            f"Avg NTP Loss: {np.mean(total_ntp_loss):.4f}, "
                            f"Avg Intuition Loss: {np.mean(total_intuition_loss):.4f}, "
                            f"Current lr: {scheduler.get_last_lr()[0]:.10f}"
                            f"Time for 10 steps: {str(datetime.timedelta(seconds=int(duration)))}", flush=True
                        )
                    else:
                        print(
                            f"Step [{(ministep + 1) // gradient_accumulation_steps}/{(len(train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps}] - "
                            f"Avg Total Loss: {np.mean(total_loss):.4f}, "
                            f"Avg NTP Loss: {np.mean(total_ntp_loss):.4f}, "
                            f"Avg Intuition Loss: {np.mean(total_intuition_loss):.4f}"
                            f"Time for 10 steps: {str(datetime.timedelta(seconds=int(duration)))}", flush=True
                        )
                step_loss = 0
                step_ntp_loss = 0
                step_intuition_loss = 0
                optimizer.zero_grad()
        print(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(total_loss):.4f} Avg NTP Loss: {np.mean(total_ntp_loss):.4f} Avg Intuition Loss: {np.mean(total_intuition_loss):.4f}", flush=True)

        print(f"Epoch [{epoch + 1}] - Start Validation", flush=True)
        model.eval()
        total_loss = []
        total_ntp_loss = []
        total_intuition_loss = []
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                intuition_tokens = batch["intuition_tokens"]
                random_tokens = batch["random_tokens"]
                tgt_tokens = batch["tgt_tokens"]
                tgt_ids = batch["tgt_ids"]
                labels = batch["labels"]

                res = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    intuition_tokens=intuition_tokens,
                    random_tokens=random_tokens,
                    tgt_tokens=tgt_tokens,
                    tgt_ids=tgt_ids,
                    labels=labels,
                )
                loss = res.loss
                ntp_loss = res.ntp_loss
                intuition_loss = res.intuition_loss
                total_loss.append(loss.item())
                total_ntp_loss.append(ntp_loss.item())
                total_intuition_loss.append(intuition_loss.item())
        print(f"Epoch [{epoch + 1}] - Validation Avg Total Loss: {np.mean(total_loss):.4f} Avg NTP Loss: {np.mean(total_ntp_loss):.4f} Avg Intuition Loss: {np.mean(total_intuition_loss):.4f}", flush=True)

        avg_valid_loss = np.mean(total_ntp_loss)
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def eval(model, tokenizer, local_batch_data, file_local_batch_data, info_file, args):


    hash_dict = get_prefix_data_inturec_f2(info_file, tokenizer)
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
    
    generation_config = GenerationConfig(
        num_beams=args.topk,
        length_penalty=args.length_penalty,
        num_return_sequences=args.topk,
        pad_token_id = model.config.pad_token_id,
        eos_token_id = model.config.eos_token_id,
        max_new_tokens = 128,
    )


    for idx, data_i in enumerate(tqdm(local_batch_data)):
        maxLen = max([len(_["input_ids"]) for _ in data_i])
        maxintuitionlen = max([
            max([
                    len(item)
                    for item in items["intuition_tokens"]
            ])
            for items in data_i
        ])

        padding_data = {"input_ids": [], "attention_mask": [], "intuition_tokens": []}

        for  _ in data_i:
            L = len(_["input_ids"])
            padding_data["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"].tolist())
            padding_data["attention_mask"].append([0] * (maxLen - L) + _["attention_mask"].tolist())
            intuition_tokens = []
            for seq in _["intuition_tokens"]:
                seq_len = len(seq)
                padded_seq = [tokenizer.pad_token_id] * (maxintuitionlen - seq_len) + seq.tolist()
                intuition_tokens.append(padded_seq)

            padding_data["intuition_tokens"].append(intuition_tokens)   


        logitsprocessor1 = TemperatureLogitsWarper(temperature=args.temperature)
        logitsprocessor2 = IntuRecLogitsProcessor(
            tokenizer=tokenizer,
            model=model,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=args.topk,
            guide_token="<|StartPredict|>",
        )
        logitsprocessor = LogitsProcessorList([logitsprocessor1, logitsprocessor2])
        generation_output = model.generate(
            input_ids=torch.tensor(padding_data["input_ids"]).to(model.device),
            attention_mask=torch.tensor(padding_data["attention_mask"]).to(model.device),
            intuition_tokens=torch.tensor(padding_data["intuition_tokens"]).to(model.device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logitsprocessor
        )
        
        startofpredict_id = tokenizer.encode("<|StartPredict|>", add_special_tokens=False)[0]
        endofpredict_id = tokenizer.encode("<|EndPredict|>", add_special_tokens=False)[0]

        mask_s1 = generation_output.sequences.eq(startofpredict_id)        # bool (bsz*beam, seq)
        assert mask_s1.sum(dim=1).eq(1).all()
        pos_s1 = mask_s1.long().argmax(dim=1)              # (bsz)

        mask_s2 = generation_output.sequences.eq(endofpredict_id)        # bool (bsz*beam, seq)
        assert mask_s2.sum(dim=1).eq(1).all()
        pos_s2 = mask_s2.long().argmax(dim=1)              # (bsz)

        arange_ids = torch.arange(mask_s1.shape[-1], device=generation_output.sequences.device).unsqueeze(0)  # (1, seq_len)
        mask_between = (arange_ids > pos_s1.unsqueeze(1)) & (arange_ids < pos_s2.unsqueeze(1))  # (bsz*beam, seq_len)
        segments = generation_output.sequences.masked_select(mask_between)  # (sum of segment lengths,)
        lengths = (mask_between.sum(dim=1)).tolist()
        segments_split = torch.split(segments, lengths)

        generation_res = tokenizer.batch_decode(segments_split, skip_special_tokens=True)
        generation_scores = generation_output.sequences_scores.tolist()

        outputs_i = [generation_res[i * args.topk: (i + 1) * args.topk] for i in range(len(generation_res) // args.topk)]
        scores_i = [generation_scores[i * args.topk: (i + 1) * args.topk] for i in range(len(generation_scores) // args.topk)]

        for i, test_row in enumerate(file_local_batch_data[idx]):
            test_row["predict"] = outputs_i[i]
            test_row["predict_score"] = scores_i[i]
            if 'dedup' in test_row:
                test_row.pop('dedup')

    data_result_meta_path = f"{args.data_result_meta_dir_path}/meta.json"

    with open(data_result_meta_path, 'w') as f:
        json.dump([test_row for batch in file_local_batch_data for test_row in batch], f, indent=4)

def run():
    args = parser_args()

    set_seed(args)
    device = get_device(args)

    args.data_result_meta_dir_path = args.result_dir or f"{args.data_result_root_path}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}/intuition_num_{args.intuition_num}/alpha_{args.alpha}/{args.decode_method}/rs_{args.reason_step}"
    os.makedirs(args.data_result_meta_dir_path, exist_ok=True)


    model, tokenizer = initial_model(args, device)

    train_loader, valid_loader, test_loader = initial_dataLoader(args, tokenizer)

    if args.train_or_test == "train":
        optimizer, scheduler = initial_optimizer_scheduler(
            model,
            lr=args.lr,
            wd=0.0,
            optimizer="adamw",
            epochs=args.nums_epoch,
            warmup_step=20,
            batchnum_per_epoch=(len(train_loader.dataset) + (args.train_batch_size - 1)) // args.train_batch_size,
            scheduler_type="linear"
        )
        train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, args)


    info_files = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", "*.txt"))
    assert len(info_files) == 1, "There should be only one txt file in the directory."
    info_file = info_files[0]

    if args.decode_method == "d3":
        args.length_penalty = 0.0
    test_data = test_loader.dataset
    raw_data = [test_data[i] for i in range(len(test_data))]
    block_num = (len(raw_data) + args.eval_batch_size - 1) // args.eval_batch_size
    batch_raw_data = [raw_data[i * args.eval_batch_size: (i + 1) * args.eval_batch_size] for i in range(block_num)]

    file_raw_data = test_data.data
    file_batch_raw_data = [file_raw_data[i * args.eval_batch_size: (i + 1) * args.eval_batch_size] for i in range(block_num)]

    eval(model, tokenizer, batch_raw_data, file_batch_raw_data, info_file, args)

    path = f"{args.data_result_meta_dir_path}/meta.json"
    
    calRecallAndNdcgFromFile(path, info_file, ans_col_name="tgt_item")
        

if __name__ == "__main__":
    run()