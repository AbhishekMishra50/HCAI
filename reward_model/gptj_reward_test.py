import random

import numpy as np
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def create_comparison_dataset(path="CarperAI/AbhishekBot/Summarize_Final_Worker_id", split="train"):
    dataset = load_dataset(path, split=split)
    if split == "test":
        dataset = dataset.select(range(5000))

    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        Chosen_summary = sample["Chosen"]
        Rejected_summary = sample["Rejected"]
        if Chosen_summary == Rejected_summary:
            continue
        if len(Chosen_summary.split()) < 5 or len(Rejected_summary.split()) < 5:
            continue
        pair["Chosen"] = prompt + "\n" + Chosen_summary
        pair["Rejected"] = prompt + "\n" + Rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.Chosen_input_ids = []
        self.Chosen_attn_masks = []
        self.Rejected_input_ids = []
        self.Rejected_attn_masks = []
        for pair in pairs:
            Chosen, Rejected = pair["Chosen"], pair["Rejected"]
            Chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + Chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            Rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + Rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(Chosen_encodings_dict["input_ids"], Rejected_encodings_dict["input_ids"])).item():
                self.Chosen_input_ids.append(Chosen_encodings_dict["input_ids"])
                self.Chosen_attn_masks.append(Chosen_encodings_dict["attention_mask"])
                self.Rejected_input_ids.append(Rejected_encodings_dict["input_ids"])
                self.Rejected_attn_masks.append(Rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.Chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.Chosen_input_ids[idx],
            self.Chosen_attn_masks[idx],
            self.Rejected_input_ids[idx],
            self.Rejected_attn_masks[idx],
        )


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
        
        
        worker_ids = torch.tensor([f[4] for f in data])
        batch["worker_ids"] = torch.cat([worker_ids, worker_ids]) #This thing need to be check
        return batch
        
        



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer(tokenizer.pad_token)["input_ids"][0]

    model = GPTRewardModel(
    "EleutherAI/gpt-neo-1.3B",
    num_workers=40,
    worker_embedding_dim=16
)
    model.load_state_dict(torch.load("/users/student/pg/pg24/abhishek/trlx/rm_checkpoint/checkpoint-39000/pytorch_model.bin"))
    max_length = 550
    val_pairs = create_comparison_dataset("AbhishekBot/Summarize_Final_Worker_id", "test")
    dev_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)

    from torch.utils.data import DataLoader

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())
    model.cuda()
    model.eval()
    model.half()
    correct = 0
    Chosen_list = []
    reject_list = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader)):
            for x in batch:
                batch[x] = batch[x].cuda()
            outputs = model(**batch)
            correct += sum(outputs["Chosen_end_scores"] > outputs["Rejected_end_scores"])
            Chosen_list.append(outputs["Chosen_end_scores"].cpu())
            reject_list.append(outputs["Rejected_end_scores"].cpu())
    print("Total accuracy: ", correct / len(dev_dataset))
