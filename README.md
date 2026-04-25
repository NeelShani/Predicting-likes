# EVOLVE-Social: Predicting and Optimizing Social Media Engagement

This repository contains the fine-tuning pipeline for **EngageBERT**, a ModernBERT-base model adapted with LoRA to predict social media engagement from post text and metadata. This is the predictive evaluator component of my MSc thesis at Czech Technical University in Prague.

## What this does

EngageBERT takes a social media post along with its author's follower count, posting year, and platform, and predicts a log-transformed like count. It is designed to serve as a zero-cost, locally executable heuristic judge inside agentic optimisation workflows.

## Key design decisions

**Implicit metadata injection over explicit normalisation**
Rather than dividing likes by followers to compute an engagement rate, follower count, year, and platform are injected directly into the text input. This lets ModernBERT's attention mechanism learn the non-linear relationship between audience size and engagement dynamically, without destroying the absolute magnitude of virality.

**Log transformation of the target variable**
Raw like counts follow a severe power-law distribution. Applying a natural log transformation compresses the long tail and allows the model to learn meaningfully from both low-performing and viral posts.

**LoRA for parameter-efficient fine-tuning**
Full fine-tuning of ModernBERT on a dataset of roughly 11,000 posts would risk catastrophic forgetting. LoRA adapters are injected into both attention and MLP modules at rank 64, reducing trainable parameters while preserving the model's pre-trained linguistic knowledge.

## Dataset

Combined Twitter and LinkedIn posts sourced from BrightData commercial datasets, totalling approximately 11,000 records after cleaning. Each record contains post text, follower count, posting year, platform label, and raw like count.

## Results

The model achieves its best performance on mid-sized accounts with 1,000 to 10,000 followers, where text quality most strongly predicts engagement. Prediction error is higher at both extremes: very small accounts introduce statistical noise, and very large accounts are driven by network effects beyond textual content.

| Audience Size | Average Error (log scale) |
|---|---|
| Under 100 followers | 1.66 |
| 1k to 10k followers | 1.32 |
| Over 100k followers | 2.16 |

## Training configuration

| Parameter | Value |
|---|---|
| Base model | ModernBERT-base |
| Fine-tuning method | LoRA |
| LoRA rank | 64 |
| Learning rate | 5e-5 |
| Batch size | 32 |
| Epochs | 5 |
| Max sequence length | 256 tokens |
| Random seed | 42 |

## Part of a larger system

EngageBERT is the evaluator component of EVOLVE-Social, a closed-loop generative optimisation framework developed as my MSc thesis. The agentic optimisation workflows that use this model as a live heuristic judge are being cleaned and documented for release alongside the thesis submission.

## Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- PEFT
- Weights and Biases (optional, for experiment tracking)

## Author

Neel Jigneshbhai Shanishvara
MSc Data Science, Czech Technical University in Prague
Supervisor: doc. Mgr. Viliam Lisý, MSc., Ph.D.
