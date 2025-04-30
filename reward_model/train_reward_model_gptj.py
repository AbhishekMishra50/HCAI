import os
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
from collections import defaultdict

# Global mapping
worker_string_to_int = {}
current_worker_int = 0

def get_worker_int_id(worker_string):
    global worker_string_to_int, current_worker_int
    if worker_string not in worker_string_to_int:
        worker_string_to_int[worker_string] = current_worker_int
        current_worker_int += 1
    return worker_string_to_int[worker_string]


def create_comparison_dataset(path="AbhishekBot/Summarize_Final_Worker_id", split="train"):
    dataset = load_dataset("AbhishekBot/Summarize_Final_Worker_id", split=split)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["Chosen"]
        rejected_summary = sample["Rejected"]
        

        worker_id_string = sample.get("worker")  #  get worker string
        worker_id = get_worker_int_id(worker_id_string)  #  map to integer

        if chosen_summary is None or rejected_summary is None:
            continue
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue

        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pair["worker_id"] = worker_id
        # print(pair)
        pairs.append(pair)
    return pairs



class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        self.worker_ids = []
        
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            worker_id = pair["worker_id"]
            
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            if not torch.all(torch.eq(chosen_encodings_dict["input_ids"], rejected_encodings_dict["input_ids"])).item():
                self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
                self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
                self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
                self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
                self.worker_ids.append(worker_id)

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
            self.worker_ids[idx],
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


def compute_metrics(eval_preds):
    chosen_end_scores = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token

    if not os.path.exists("rm_checkpoint"):
        os.mkdir("rm_checkpoint")

    training_args = TrainingArguments(
        output_dir="rm_checkpoint/",
        num_train_epochs=2,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=1000,
        save_steps=1000,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        # deepspeed="ds_config_gpt_j.json",
        save_total_limit=1,
    )

    # Initialize the reward model with worker embeddings
    # Estimate the number of workers in your dataset or set a reasonable upper bound
    num_workers = 40  # Adjust based on your dataset
    model = GPTRewardModel("EleutherAI/gpt-neo-1.3B", num_workers=num_workers, worker_embedding_dim=16)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.device = device  

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    # Create the comparisons datasets
    data_path = "AbhishekBot/Summarize_Final_Worker_id"
    train_pairs = create_comparison_dataset(data_path, "train")
    print(f"Train raw dataset: {len(load_dataset(data_path, split='train'))}")
    print(f"Train after filtering: {len(train_pairs)}")
    val_pairs = create_comparison_dataset(data_path, "test")
    print(f"Test raw dataset: {len(load_dataset(data_path, split='test'))}")
    print(f"Test after filtering: {len(val_pairs)}")


    # Make pairwise datasets for training
    max_length = 550
    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_length)
    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Example worker_ids (first 5 examples): {[train_dataset[i][4] for i in range(5)]}")

    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_length)
    print(f"Number of validation examples: {len(val_dataset)}")
    print(f"Example worker_ids (first 5 examples): {[val_dataset[i][4] for i in range(5)]}")
    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()
    import pickle
    with open("worker_string_to_int.pkl", "wb") as f:
        pickle.dump(worker_string_to_int, f)
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    ).train()
