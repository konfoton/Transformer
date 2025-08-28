import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken
import config


def gen_all_tokens(raw, encoding, eos_id):
    for item in raw:
        txt = item["text"].strip()
        if not txt:
            continue
        tokens = encoding.encode(txt)
        tokens.append(eos_id)
        yield from tokens

class LMChunks(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    def __len__(self):
        return self.inputs.size(0)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    

if __name__ == "__main__":
    encoding = tiktoken.get_encoding("gpt2")
    eos_id = encoding.eot_token  # 50256
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    all_tokens = list(gen_all_tokens(raw, encoding, eos_id))
    data = torch.tensor(all_tokens, dtype=torch.long)
    num_blocks = (len(data) - 1) // config.TrainingConfig.context_window
    data = data[: num_blocks * config.TrainingConfig.context_window + 1] 
    inputs = data[:-1].view(num_blocks, config.TrainingConfig.context_window)
    targets = data[1:].view(num_blocks, config.TrainingConfig.context_window)
    dataset = LMChunks(inputs, targets)
    torch.save(dataset, "wikitext103_tokenized.pt")
