# Transformer Model with LoRA Fine-Tuning

This repository contains a Transformer model implementation following the architecture and training methodology described in the **“Attention is All You Need”** paper. Additionally, the model has been fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient adaptation.

## Training
Training was performed on 2x A100 PCIe GPUs and each update was performed on  16 384 tokens (GPT2 tokenizer)

Below is a summary of the training loss progression during pre-training (one step is 50 updates):

<img width="5056" height="2656" alt="W B Chart 28_08_2025, 15_47_27" src="https://github.com/user-attachments/assets/fe3ad313-33bf-4e67-9ef5-fc56beb7d013" />
