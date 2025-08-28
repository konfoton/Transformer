from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size = 50257
    n_layer = 6
    n_head = 8
    n_embd = 512
    key_query_second_dim = 64
    value_second_dim = 64
    d_mlp = 2048
    dropout = 0.1

@dataclass
class TrainingConfig:
    global_batch_size = 32
    adam_beta = 0.9
    adam_beta2 = 0.95
    eps = 1e-9
    warmup_steps = 4000
    context_window = 512
    epochs = 100
    grad_clip = 1.0
    eval_interval = 2000