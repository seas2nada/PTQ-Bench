import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

import torch
import random
from datasets import load_dataset

def format_humaneval(ex):
    # HumanEval has "prompt" which contains function signature and docstring
    prompt = ex["prompt"].strip()
    # Add instruction to write code
    return f"Write a Python function to solve the following problem:\n{prompt}\nSolution:"

def get_humaneval(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    # HumanEval only has "test" split
    split = "test"
    ds = load_dataset("openai_humaneval", split=split)
    
    if eval_mode:
        return ds
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_humaneval(ds[idx]), return_tensors="pt")
        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i+seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def format_medqa(ex):
    # MedQA example has "question", "options", "answer_idx"
    question = ex["question"].strip()
    options = ex["choices"]
    # options might be a list of strings
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    return f"Question: {question}\nOptions:\n{options_str}\nAnswer:"

def get_medqa(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    # Use "medqa" dataset, assuming splits: train, validation, test
    split = "train" if not eval_mode else "test"
    try:
        ds = load_dataset("medqa", split=split)
    except:
        # fallback to bigbio/med_qa
        ds = load_dataset("bigbio/med_qa", name="med_qa_en_bigbio_qa", split=split)
    if eval_mode:
        return ds
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_medqa(ds[idx]), return_tensors="pt")
        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i+seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def format_truthfulqa(ex):
    # TruthfulQA generation: has "question"
    question = ex["question"].strip()
    return f"Question: {question}\nAnswer:"

def get_truthfulqa(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    # TruthfulQA generation subset has only "validation" split
    split = "validation"
    ds = load_dataset("truthful_qa", "generation", split=split)
    if eval_mode:
        return ds
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_truthfulqa(ds[idx]), return_tensors="pt")
        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i+seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def format_piqa(ex):
    goal = ex["goal"].strip()
    sol1 = ex["sol1"].strip()
    sol2 = ex["sol2"].strip()
    # PIQA는 2-choice라 A/B로 두는 게 깔끔
    return (
        f"Goal: {goal}\n"
        f"Choices:\n"
        f"A. {sol1}\n"
        f"B. {sol2}\n"
        f"Answer:"
    )

def format_winograde(ex):
    # winogrande: sentence에 '_' 빈칸이 있고 option1/2 중 하나로 채움
    sent = ex["sentence"].strip()
    opt1 = ex["option1"].strip()
    opt2 = ex["option2"].strip()
    return (
        f"Sentence: {sent}\n"
        f"Choices:\n"
        f"A. {opt1}\n"
        f"B. {opt2}\n"
        f"Answer:"
    )

def format_boolq(ex):
    passage = ex["passage"].strip()
    question = ex["question"].strip()

    return (
        f"Passage: {passage}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

def format_arc(ex):
    q = ex["question"].strip()
    texts = ex["choices"]["text"]
    labels = ex["choices"]["label"]

    opts = "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])
    return f"Question: {q}\nChoices:\n{opts}\nAnswer:"

def format_hellaswag(ex):
    ctx = (ex["ctx_a"] + " " + ex["ctx_b"]).strip()
    endings = ex["endings"]
    labels = ["A", "B", "C", "D"]

    opts = "\n".join([f"{labels[i]}. {endings[i]}" for i in range(4)])
    return f"Context: {ctx}\nEndings:\n{opts}\nAnswer:"

def format_gsm8k(ex):
    question = ex["question"].strip()
    answer = ex["answer"].strip()
    return f"Question: {question}\nAnswer:"

def get_pile(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_pile")
    traindata = load_dataset("json", data_files='/path/to/val.jsonl.zst', split="train")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_wikitext2")
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    if tokenizer is None:
        print(model)
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc 

def get_ptb_new(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata  = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata ["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model, tokenizer=None):
    print("get_c4_new")
    traindata = load_dataset(
        'json', data_files={'train': '/path/to/c4/en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'json', data_files={'validation': '/path/to/c4/en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc

def get_arc(nsamples, seqlen, tokenizer, seed=0, eval_mode=False, subset="challenge"):
    split = "train" if not eval_mode else "validation"
    random.seed(seed)

    subset = subset.lower()
    config = "ARC-Challenge" if subset in ["challenge", "arc-challenge"] else "ARC-Easy"

    ds = load_dataset("ai2_arc", config, split=split)

    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_arc(ds[idx]), return_tensors="pt")

        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_hellaswag(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    split = "train" if not eval_mode else "validation"
    ds = load_dataset("hellaswag", split=split)

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_hellaswag(ds[idx]), return_tensors="pt")

        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_boolq(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    split = "train" if not eval_mode else "validation"
    ds = load_dataset("boolq", split=split)

    if eval_mode:
        return ds  # 🔥 텐서 말고 raw dataset 반환

    random.seed(seed)
    # (adapt/trainloader는 기존처럼 유지)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_boolq(ds[idx]), return_tensors="pt")
        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def get_winogrande(nsamples, seqlen, tokenizer, seed=0, eval_mode=False, config="winogrande_xl"):
    # HF: dataset name은 "winogrande", config는 보통 "winogrande_xl"을 많이 씀
    split = "train" if not eval_mode else "validation"
    ds = load_dataset("winogrande", config, split=split)

    if eval_mode:
        # accuracy_eval에서 sentence/option1/option2/answer를 쓰기 좋게 raw 그대로 반환
        return ds

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_winograde(ds[idx]), return_tensors="pt")

        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_piqa(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    split = "train" if not eval_mode else "validation"
    ds = load_dataset("baber/piqa", split="train", trust_remote_code=True)

    if eval_mode:
        # accuracy_eval에서 goal/sol1/sol2/label 그대로 쓰는 게 제일 편함
        return ds

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_piqa(ds[idx]), return_tensors="pt")

        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_gsm8k(nsamples, seqlen, tokenizer, seed=0, eval_mode=False):
    split = "train" if not eval_mode else "validation"
    ds = load_dataset("gsm8k", "main", split=split)

    if eval_mode:
        return ds

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        idx = random.randint(0, len(ds) - 1)
        enc = tokenizer(format_gsm8k(ds[idx]), return_tensors="pt")

        L = enc.input_ids.shape[1]
        if L >= seqlen:
            i = random.randint(0, L - seqlen)
            inp = enc.input_ids[:, i:i + seqlen]
        else:
            inp = torch.nn.functional.pad(
                enc.input_ids,
                (0, seqlen - L),
                value=tokenizer.pad_token_id or 0,
            )

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', tokenizer=None
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
        
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'pile' in name:
        return get_pile(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, model, tokenizer)  
        return get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, model, tokenizer)  
        return get_c4(nsamples, seed, seqlen, model, tokenizer)
    if name.lower() == "arc_c":
        data = get_arc(nsamples, seqlen, tokenizer, seed=seed, subset="challenge")
    if name.lower() == "arc_e":
        data = get_arc(nsamples, seqlen, tokenizer, seed=seed, subset="easy")
    if name.lower() == "hellaswag":
        data = get_hellaswag(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "boolq":
        data = get_boolq(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "piqa":
        data = get_piqa(nsamples, seqlen, tokenizer, seed=seed)      
    if name.lower() == "winogrande":
        data = get_winogrande(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "gsm8k":
        data = get_gsm8k(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "humaneval":
        data = get_humaneval(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "medqa":
        data = get_medqa(nsamples, seqlen, tokenizer, seed=seed)
    if name.lower() == "truthfulqa":
        data = get_truthfulqa(nsamples, seqlen, tokenizer, seed=seed)
    
    if name.lower() in ["arc_c", "arc_e", "hellaswag", "boolq", "piqa", "winogrande", "gsm8k", "humaneval", "medqa", "truthfulqa"]:
        return data, None
    if name.lower() == "mix":
        wiki_train,wiki_val=get_wikitext2(nsamples//4, seed, seqlen, model, tokenizer)
        boolq_train=get_boolq(nsamples//4, seqlen, tokenizer, seed=seed)
        piqa_train=get_piqa(nsamples//4, seqlen, tokenizer, seed=seed)
        winogrande_train=get_winogrande(nsamples//4, seqlen, tokenizer, seed=seed)
        train=wiki_train+boolq_train+piqa_train+winogrande_train
        val=None
        return train,val
    elif name.lower() == "mix128":
        wiki_train,wiki_val=get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
        boolq_train=get_boolq(nsamples, seqlen, tokenizer, seed=seed)
        piqa_train=get_piqa(nsamples, seqlen, tokenizer, seed=seed)
        winogrande_train=get_winogrande(nsamples, seqlen, tokenizer, seed=seed)
        train=wiki_train+boolq_train+piqa_train+winogrande_train
        val=None
        return train,val