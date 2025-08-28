import math
import torch
import torch.nn as nn
from block import Block
import torch.nn.functional as F



class Model(nn.Module):
    
    def __init__(self, embed_dim, num_heads, d_mlp, key_query_second_dim, value_second_dim, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_mlp = d_mlp
        self.key_query_second_dim = key_query_second_dim
        self.value_second_dim = value_second_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        

        self.Transformer = nn.ModuleList([
            Block(embed_dim, num_heads, d_mlp, key_query_second_dim, value_second_dim) 
            for _ in range(6)
        ])

        
            


    def positional_encoding(self, x):
        B, L, E = x.size()
    
        position = torch.arange(0, L, dtype=torch.float32, device=x.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, E, 2, dtype=torch.float32, device=x.device) * 
                            -(math.log(10000.0) / E))
        
        pe = torch.zeros(L, E, device=x.device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).expand(B, -1, -1)
        
        return x + pe 
        

    def forward(self, x, target=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for block in self.Transformer:
            x = block(x)
        logits = F.linear(x, self.embedding.weight)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), target.view(-1))
        return logits, loss


    def generate_k_tokens(self, x, k):
        generated = []
        for _ in range(k):
            logits, _ = self.forward(x)              # (B, current_len, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(next_token)
            x = torch.cat([x, next_token], dim=1)  
        return torch.cat(generated, dim=1)
    
    def add_lora(self, lora, lora_state_dict=None):
        for block in self.Transformer:
            block.cross_attention.key = lora.wrap(block.cross_attention.key, 'key')
            block.cross_attention.query = lora.wrap(block.cross_attention.query, 'query')
            block.cross_attention.value = lora.wrap(block.cross_attention.value, 'value')
            block.cross_attention.output = lora.wrap(block.cross_attention.output, 'output')
            
        
        if lora_state_dict is not None:
            self.load_lora_state_dict(lora_state_dict)
            
        lora.mark_only_lora_trainable(self)
    
    @classmethod
    def load_from_pretrained(cls, checkpoint_path, embed_dim, num_heads, d_mlp, 
                           key_query_second_dim, value_second_dim, vocab_size, 
                           device='cpu', strict=True):
        """
        Load model from your saved checkpoint format
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model = cls(embed_dim, num_heads, d_mlp, key_query_second_dim, 
                   value_second_dim, vocab_size)
        
        state_dict = checkpoint["model_state"]
        
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()
        
        checkpoint_info = {
            'step': checkpoint.get('step', 0),
            'best_val': checkpoint.get('best_val', float('inf')),
            'optimizer_state': checkpoint.get('optimizer_state', None)
        }
        
        return model, checkpoint_info
    

    def save_lora_state_dict(self, path):
        lora_state = {}
        for name, param in self.named_parameters():
            if 'A' in name or 'B' in name:
                lora_state[name] = param.data.clone()
        torch.save(lora_state, path)
        print(f"LoRA weights saved to {path}")
    

    def load_lora_state_dict(self, lora_state_dict_or_path):
        if isinstance(lora_state_dict_or_path, str):
            lora_state_dict = torch.load(lora_state_dict_or_path, map_location=next(self.parameters()).device)
        else:
            lora_state_dict = lora_state_dict_or_path
        
        model_state = self.state_dict()
        for name, param in lora_state_dict.items():
            if name in model_state and ('A' in name or 'B' in name):
                model_state[name].copy_(param)
                print(f"Loaded LoRA parameter: {name}")
        
        print(f"Loaded {len(lora_state_dict)} LoRA parameters")
