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
from accelerate import Accelerator
from accelerate.utils import set_seed, TorchDynamoPlugin


def lr_schedule(step, d_model, warmup):
    if step == 0:
        step = 1
    scale = d_model ** -0.5
    return scale * min(step ** -0.5, step * (warmup ** -1.5))


def evaluate(accelerator, model, val_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            _, loss = model(x, y)
            losses.append(loss)
    loss_tensor = torch.stack(losses)
    all_losses = accelerator.gather_for_metrics(loss_tensor)
    val_loss = all_losses.mean().item()
    ppl = math.exp(val_loss)
    model.train()
    return val_loss, ppl


def sample(accelerator, model, prefix_ids, max_new=30):
    if not accelerator.is_main_process:
        return None
    with torch.no_grad():
        x = torch.tensor([prefix_ids], dtype=torch.long, device=accelerator.device)
        out = accelerator.unwrap_model(model).generate_k_tokens(x, max_new)
        return out[0].tolist()


def load_checkpoint_if_exists(path, model, optimizer):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return ckpt.get("step", 0), ckpt.get("best_val", float("inf"))
    return 0, float("inf")


def main():
    set_seed(42)
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    dataset = torch.load("wikitext103_tokenized.pt", weights_only=False)

    val_frac = 0.01
    val_len = int(len(dataset) * val_frac)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.global_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.global_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model = Model(
        embed_dim=model_cfg.n_embd,
        num_heads=model_cfg.n_head,
        d_mlp=model_cfg.d_mlp,
        key_query_second_dim=model_cfg.key_query_second_dim,
        value_second_dim=model_cfg.value_second_dim,
        vocab_size=model_cfg.vocab_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0, 
        betas=(train_cfg.adam_beta, train_cfg.adam_beta2),
        eps=train_cfg.eps,
    )

    start_step, best_val = load_checkpoint_if_exists("checkpoint.pt", model, optimizer)



    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device  

    (
        train_loader,
        val_loader,
        model,
        optimizer,
    ) = accelerator.prepare(train_loader, val_loader, model, optimizer)

    param_count = sum(p.numel() for p in model.parameters())

    if accelerator.is_main_process:
        wandb.init(
            project="transformer",
            config={
                "batch_size": train_cfg.global_batch_size,
                "num_epochs": train_cfg.epochs,
                "context_window": train_cfg.context_window,
                "n_layer": model_cfg.n_layer,
                "param_count": param_count,
            },
        )

    warmup = train_cfg.warmup_steps
    d_model = model_cfg.n_embd
    global_step = start_step
    eval_interval = train_cfg.eval_interval

    model.train()

    num_epochs = train_cfg.epochs

    for epoch in range(num_epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            global_step += 1
            lr = lr_schedule(global_step, d_model, warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            _, loss = model(x, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                if global_step % 50 == 0:
                    wandb.log({"train_loss": loss.item(), "lr": lr, "step": global_step})
                    print(
                        f"epoch {epoch} step {global_step} lr {lr:.3e} loss {loss.item():.4f}"
                    )

            if global_step % eval_interval == 0:
                val_loss, val_ppl = evaluate(accelerator, model, val_loader)
                if accelerator.is_main_process:
                    wandb.log(
                        {"val_loss": val_loss, "val_ppl": val_ppl, "step": global_step}
                    )
                    print(
                        f"[val] step {global_step} loss {val_loss:.4f} ppl {val_ppl:.2f}"
                    )
                    if val_loss < best_val:
                        best_val = val_loss
                        accelerator.save(
                            {
                                "model_state": accelerator.unwrap_model(model).state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "step": global_step,
                                "best_val": best_val,
                            },
                            "best_checkpoint.pt",
                        )

                    sample_ids = sample(accelerator, model, [220], max_new=20)
                    if sample_ids:
                        try:
                            tokenizer = tiktoken.get_encoding("gpt2")
                            enc = tokenizer.decode(sample_ids)
                            print(enc)
                        except Exception:
                            pass

                if accelerator.is_main_process:
                    accelerator.save(
                        {
                            "model_state": accelerator.unwrap_model(model).state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "step": global_step,
                            "best_val": best_val,
                        },
                        "checkpoint.pt",
                    )

    if accelerator.is_main_process:
        print("Training complete.")


if __name__ == "__main__":
    main()