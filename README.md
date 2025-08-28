# Transformer Model with LoRA Fine-Tuning

This repository contains a Transformer model implementation following the architecture and training methodology described in the **“Attention is All You Need”** paper. Additionally, the model has been fine-tuned using **LoRA (Low-Rank Adaptation)** for parameter-efficient adaptation.

## Training
Training was performed on 2x A100 PCIe GPUs and each update was performed on  16 384 tokens (GPT2 tokenizer)

Unfortunately due to insufficinet funds trainnig lasted only 30minutes and costed 2 dollars

I used mixed precision learning especially bfloat16 for training and float32 for stored weights

Below is a summary of the training loss progression during pre-training (one step is 50 updates):

<img width="500" height="250" alt="W B Chart 28_08_2025, 15_56_35" src="https://github.com/user-attachments/assets/c4003d5b-df87-4031-b101-b815fcb246b4" />
