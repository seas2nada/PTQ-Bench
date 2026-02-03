import torch
from datasets import load_dataset

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

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == 'c4':
        dataset = load_dataset(
        'json', data_files={'train': '/path/to/c4/en/c4-train.00000-of-01024.json'}, split='train'
    )
    elif data == 'wikitext2':
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    elif data == 'boolq':
        dataset = load_dataset('boolq', split='train')
        dataset = dataset.map(lambda ex: {"text": format_boolq(ex)}, remove_columns=dataset.column_names)
    elif data == 'piqa':
        dataset = load_dataset('piqa', split='train')
        dataset = dataset.map(lambda ex: {"text": format_piqa(ex)}, remove_columns=dataset.column_names)
    elif data == 'winogrande':
        dataset = load_dataset('winogrande', 'winogrande_xl', split='train')
        dataset = dataset.map(lambda ex: {"text": format_winograde(ex)}, remove_columns=dataset.column_names)
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
