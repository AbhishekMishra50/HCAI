import random
import numpy as np
import torch
from datasets import load_dataset
from reward_model import GPTRewardModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# --- Set random seeds ---6,
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

# --- Load saved worker_string_to_int mapping ---
def load_worker_mapping(mapping_path):
    with open(mapping_path, "rb") as f:
        worker_string_to_int = pickle.load(f)
    return worker_string_to_int

# --- Dataset preparation ---
def create_comparison_dataset(path="AbhishekBot/Summarize_Final_Worker_id", split="test", worker_string_to_int=None):
    dataset = load_dataset(path, split="test")
    if split == "test":
        dataset = dataset.select(range(5000))  # optional, remove if not needed

    pairs = []
    for sample in tqdm(dataset, desc=f"Loading {split} set"):
        pair = {}
        prompt = sample["prompt"]
        Chosen_summary = sample["Chosen"]
        Rejected_summary = sample["Rejected"]
        worker_id_string = sample.get("worker")

        if worker_id_string is None or worker_string_to_int is None:
            continue

        # Use pre-loaded worker mapping
        worker_id = worker_string_to_int.get(worker_id_string, 0)  # map unknown workers to ID 0

        if Chosen_summary == Rejected_summary:
            continue
        if len(Chosen_summary.split()) < 5 or len(Rejected_summary.split()) < 5:
            continue

        pair["Chosen"] = prompt + "\n" + Chosen_summary
        pair["Rejected"] = prompt + "\n" + Rejected_summary
        pair["worker_id"] = worker_id

        pairs.append(pair)
    
    return pairs

class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.Chosen_input_ids = []
        self.Chosen_attn_masks = []
        self.Rejected_input_ids = []
        self.Rejected_attn_masks = []
        self.worker_ids = []

        for pair in tqdm(pairs, desc="Tokenizing pairs"):
            Chosen, Rejected = pair["Chosen"], pair["Rejected"]
            worker_id = pair["worker_id"]

            Chosen_encodings = tokenizer(
                "<|startoftext|>" + Chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            Rejected_encodings = tokenizer(
                "<|startoftext|>" + Rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            if not torch.all(torch.eq(Chosen_encodings["input_ids"], Rejected_encodings["input_ids"])).item():
                self.Chosen_input_ids.append(Chosen_encodings["input_ids"])
                self.Chosen_attn_masks.append(Chosen_encodings["attention_mask"])
                self.Rejected_input_ids.append(Rejected_encodings["input_ids"])
                self.Rejected_attn_masks.append(Rejected_encodings["attention_mask"])
                self.worker_ids.append(worker_id)

    def __len__(self):
        return len(self.Chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.Chosen_input_ids[idx],
            self.Chosen_attn_masks[idx],
            self.Rejected_input_ids[idx],
            self.Rejected_attn_masks[idx],
            self.worker_ids[idx],
        )

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
        batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
        batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))

        worker_ids = torch.tensor([f[4] for f in data])
        batch["worker_ids"] = torch.cat([worker_ids, worker_ids])  # Double because inputs are doubled
        
        return batch

# --- Main Evaluation ---
if __name__ == "__main__":
    set_seed(42)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    tokenizer.pad_token = tokenizer.eos_token

    # Load worker id mapping (VERY IMPORTANT!)
    mapping_path = "/users/student/pg/pg24/abhishek/trlx/worker_string_to_int.pkl"  # <<<--- Change this to your actual file
    assert os.path.exists(mapping_path), "Mapping file not found!"
    worker_string_to_int = load_worker_mapping(mapping_path)

    # Load model
    model = GPTRewardModel(
        "EleutherAI/gpt-neo-1.3B",
        num_workers=40,
        worker_embedding_dim=16
    )
    model.load_state_dict(torch.load("/users/student/pg/pg24/abhishek/trlx/rm_checkpoint/checkpoint-39000/pytorch_model.bin"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.half()

    # Prepare dataset
    val_pairs = create_comparison_dataset(
        path="AbhishekBot/Summarize_Final_Worker_id",
        split="test",
        worker_string_to_int=worker_string_to_int
    )
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=550)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=6, collate_fn=DataCollatorReward())

    # Evaluate
    correct = 0
    total = len(val_dataset)
    Chosen_scores = []
    Rejected_scores = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Evaluating"):
            for x in batch:
                batch[x] = batch[x].to(device)

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                worker_ids=batch["worker_ids"]
            )

            correct += (outputs["chosen_end_scores"] > outputs["rejected_end_scores"]).sum().item()
            Chosen_scores.append(outputs["chosen_end_scores"].cpu())
            Rejected_scores.append(outputs["rejected_end_scores"].cpu())

    # Final results
    print(Chosen_scores[0][0].item())
    print(Rejected_scores[0][0].item())
    accuracy = correct / total
    print(f"\nTotal accuracy on test set (with correct worker embeddings): {accuracy:.4f}")
