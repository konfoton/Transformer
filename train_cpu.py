import os
import math
import random
import torch
import wandb
from torch.utils.data import DataLoader, random_split
from config import ModelConfig, TrainingConfig
from model import Model
import tiktoken
from tokenize_dataset import LMChunks



def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

dataset = torch.load("wikitext103_tokenized.pt", weights_only=False)

val_frac = 0.01
val_len = int(len(dataset) * val_frac)
train_len = len(dataset) - val_len
train_ds, val_ds = random_split(dataset, [train_len, val_len])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cfg = ModelConfig()
train_cfg = TrainingConfig()

model = Model(
    embed_dim=model_cfg.n_embd,
    num_heads=model_cfg.n_head,
    d_mlp=model_cfg.d_mlp,
    key_query_second_dim=model_cfg.key_query_second_dim,
    value_second_dim=model_cfg.value_second_dim,
    vocab_size=model_cfg.vocab_size,
).to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
        print("Compiled with torch.compile")
    except Exception as e:
        print(f"torch.compile skipped: {e}")

param_count = sum(p.numel() for p in model.parameters())

wandb.init(project="transformer", config={
    "batch_size": train_cfg.batch_size,
    "num_epochs": train_cfg.num_epochs,
    "context_window": train_cfg.context_window,
    "n_layer": model_cfg.n_layer,
    "param_count": param_count
})

train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=False, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=1.0,
                              betas=(train_cfg.adam_beta, train_cfg.adam_beta2),
                              eps=train_cfg.eps,)


optimizer.zero_grad(set_to_none=True)
warmup = train_cfg.warmup_steps
d_model = model_cfg.n_embd
global_step = 0
best_val = float("inf")

grad_clip = 1.0
accum_steps = 1  # increase for gradient accumulation

def lr_schedule(step):
    if step == 0:
        step = 1
    scale = d_model ** -0.5
    return scale * min(step ** -0.5, step * (warmup ** -1.5))

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    val_loss = sum(losses) / len(losses)
    ppl = math.exp(val_loss)
    return val_loss, ppl

def sample(prefix_ids, max_new=30):
    with torch.no_grad():
        x = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        out = model.generate_k_tokens(x, max_new)
        return out[0].tolist()

if os.path.exists("checkpoint.pt"):
    ckpt = torch.load("checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    global_step = ckpt.get("step", 0)
    best_val = ckpt.get("best_val", best_val)
    print(f"Resumed from step {global_step}")

for epoch in range(train_cfg.num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        global_step += 1
        lr = lr_schedule(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _, loss = model(x, y)
        loss = loss / accum_steps
        loss.backward()


        wandb.log({"train_loss": loss.item() * accum_steps, "lr": lr, "step": global_step})
        print(f"epoch {epoch} step {global_step} lr {lr:.3e} loss {loss.item()*accum_steps:.4f}")

        if global_step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if global_step % 500 == 0:
            val_loss, val_ppl = evaluate()
            wandb.log({"val_loss": val_loss, "val_ppl": val_ppl, "step": global_step})
            print(f"[val] step {global_step} loss {val_loss:.4f} ppl {val_ppl:.2f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "step": global_step,
                            "best_val": best_val},
                           "best_checkpoint.pt")
            try:
                sample_ids = sample([220], max_new=20)
                tokenizer = tiktoken.get_encoding("gpt2")
                enc = tokenizer.decode(sample_ids)
                print(enc)
            except Exception:
                pass
