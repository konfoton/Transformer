# Transformer Model with LoRA Fine-Tuning

This repository contains a Transformer model implementation following the architecture and training methodology described in the **“Attention is All You Need”** paper. Additionally, the model has been fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient adaptation.

## Training

The model was trained using the standard Transformer setup with:

- Attention-based layers  
- Positional encoding  
- Multi-head self-attention  

LoRA was applied to selectively fine-tune the model, reducing the number of trainable parameters while maintaining performance.

## Training Loss

Below is a summary of the training loss progression during pre-training:

