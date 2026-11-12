import argparse
import os
import shutil
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    GenerationConfig,
    LogitsProcessorList, 
    TemperatureLogitsWarper
)
import torch
from data.IntuRecF1Dataset import IntuRecF1Dataset
from datasets import Dataset
from model.IntuRecF1 import IntuRecF1
import glob
from tqdm import tqdm
import json
from torch.distributed import barrier, is_initialized
from utils.util import get_hash, get_prefix_data_inturec_f1, calRecallAndNdcgFromFile
from model.LogitsProcessor import IntuRecLogitsProcessor

MODEL_NAME="IntuRecF1"

def parser_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)

    parser.add_argument("--seed", type=int, default=0)


    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--dataset", choices=["CDs_and_Vinyl", "Toys_and_Games", "Video_Games", "Musical_Instruments", "Books"], default="CDs_and_Vinyl")
    parser.add_argument("--sample", type=int, default=-1)

    # Please fill in according to your configuration.
    parser.add_argument("--base_model", type=str, default="your llm model path")

    parser.add_argument("--nums_epoch", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--train_micro_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--cutoff_len", type=int, default=512)

    # Please fill in according to your configuration.
    parser.add_argument("--output_dir", type=str, default=f"checkpoints/")

    parser.add_argument("--lr", type=float, default=3e-4)
    
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


def train(model, tokenizer, train_dataset, valid_dataset, args):
    model.train()
    output_dir = args.ckpt_dir or f"{args.output_dir}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}"
        

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=args.train_micro_batch_size,
            per_device_eval_batch_size=args.train_micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=args.nums_epoch,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=True,
            report_to='none',
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=1)
        ]
    )
    trainer.train()

    if torch.distributed.get_rank() == 0:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def eval(model, tokenizer, local_batch_data, file_local_batch_data, info_file, args, rank=0):


    hash_dict = get_prefix_data_inturec_f1(info_file, tokenizer)
    
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
            num_beams=args.topk,
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

    data_result_meta_path = f"{args.data_result_meta_dir_path}/meta_rank_{rank}.json"

    with open(data_result_meta_path, 'w') as f:
        json.dump([test_row for batch in file_local_batch_data for test_row in batch], f, indent=4)


def initial_model(args, device_map):
    if args.train_or_test == "test":
        args.base_model = args.ckpt_dir or f"{args.output_dir}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}"
    else:
        args.base_model = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True) 
    if args.train_or_test == "train":
        additional_special_tokens = tokenizer.additional_special_tokens
        additional_special_tokens.append("<|StartPredict|>")
        additional_special_tokens.append("<|EndPredict|>")
        tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        tokenizer.padding_side = "left"

    model = IntuRecF1.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    if args.train_or_test == "train":
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EndPredict|>")
        model.config.use_cache = False
    return model, tokenizer

def run():
    args = parser_args()

    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # assert world_size != 1
    gradient_accumulation_steps = args.train_batch_size // args.train_micro_batch_size // world_size
    args.gradient_accumulation_steps = gradient_accumulation_steps
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    args.data_result_meta_dir_path = args.result_dir or f"{args.data_result_root_path}/{args.dataset}/{MODEL_NAME}/lr_{args.lr}/{args.decode_method}"
    os.makedirs(args.data_result_meta_dir_path, exist_ok=True)

    model, tokenizer = initial_model(args, device_map)


    train_data = IntuRecF1Dataset(
        args.data_path, 
        args.dataset, 
        tokenizer, 
        sample=args.sample,
        dataset_type="train",
        seed=args.seed
    )
    valid_data = IntuRecF1Dataset(
        args.data_path, 
        args.dataset, 
        tokenizer, 
        sample=args.sample,
        dataset_type="valid",
        seed=args.seed
    )
    test_data = IntuRecF1Dataset(
        args.data_path, 
        args.dataset, 
        tokenizer, 
        sample=args.sample,
        dataset_type="test",
        seed=args.seed
    )

    print(f"Train dataset sample: \n{train_data.text_data[0]['all_input']}\n")
    print(f"Valid dataset sample: \n{valid_data.text_data[0]['all_input']}\n")
    print(f"Test dataset sample: \n{test_data.text_data[0]['all_input']}\n")

    train_dataset = Dataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    valid_dataset = Dataset.from_dict({k: [v[k] for v in valid_data] for k in valid_data[0].keys()})


    info_files = glob.glob(os.path.join(f"{args.data_path}/{args.dataset}/info", f"{args.dataset}*.txt"))
    assert len(info_files) == 1, "There should be only one txt file in the directory."
    info_file = info_files[0]

    if args.train_or_test == "train":
        train(model, tokenizer, train_dataset, valid_dataset, args)

    if args.decode_method == "d3":
        args.length_penalty = 0.0
    raw_data = [test_data[i] for i in range(len(test_data))]
    block_num = (len(raw_data) + args.eval_batch_size - 1) // args.eval_batch_size
    batch_raw_data = [raw_data[i * args.eval_batch_size: (i + 1) * args.eval_batch_size] for i in range(block_num)]
    local_batch_data = batch_raw_data[device_map[""]::world_size]

    file_raw_data = test_data.data
    file_batch_raw_data = [file_raw_data[i * args.eval_batch_size: (i + 1) * args.eval_batch_size] for i in range(block_num)]
    file_local_batch_data = file_batch_raw_data[device_map[""]::world_size]

    if is_initialized():
        barrier()
    else:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        current_device = torch.device(f"cuda:{local_rank}")
        
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=current_device
        )
        barrier()
    eval(model, tokenizer, local_batch_data, file_local_batch_data, info_file, args, rank=device_map[""])

    if is_initialized():
        barrier()
    if device_map[""] == 0:
        merged_data = []

        for r in range(world_size):
            path = f"{args.data_result_meta_dir_path}/meta_rank_{r}.json"
            assert os.path.exists(path), f"File {path} does not exist."
            with open(path, "r") as f:
                data = json.load(f)
                merged_data.extend(data)

        merged_path = os.path.join(args.data_result_meta_dir_path, "meta_merged.json")
        with open(merged_path, "w") as f:
            json.dump(merged_data, f, indent=4)
        
        calRecallAndNdcgFromFile(merged_path, info_file, ans_col_name="tgt_item")

if __name__ == "__main__":
    run()