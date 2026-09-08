# Diagnosing Optimization During Fine-Tuning

This project compares **AdamW** and **Muon** while fine-tuning a small pretrained language model on the **SST-2 sentiment classification dataset**.

The goal is not just to compare final accuracy, but to understand how the two optimizers behave during training.

## What I did

I fine-tuned `prajjwal1/bert-tiny` on SST-2 using a manual PyTorch training and evaluation loop.

For the Muon setup, Muon is used only on eligible 2D transformer weight matrices. AdamW is used for the remaining parameters such as embeddings, biases, LayerNorm parameters, and the classification head.

Both optimizers were trained under the same general setup:

- Model: `prajjwal1/bert-tiny`
- Dataset: GLUE SST-2
- Epochs: 3
- Batch size: 32
- Max sequence length: 128
- Hardware: 1 NVIDIA T4 GPU
- Seeds: 42, 43, 44
- Linear learning-rate schedule with 10% warmup
- Gradient clipping at 1.0

A small learning-rate search was performed separately for AdamW and Muon before the final runs.

## Metrics

I recorded several metrics to understand both performance and optimization behavior:

- **Training loss**  
  Shows how quickly the optimizer reduces the training objective.

- **Validation loss**  
  Used as the main comparison metric because it captures both correctness and prediction confidence.

- **Validation accuracy**  
  Measures sentiment-classification performance.

- **Gradient norm**  
  Helps show how large or unstable the gradients are during training.

- **Relative parameter update**  
  Measures how much the optimizer changes the model parameters relative to their current size.

- **Runtime**  
  Gives a rough comparison of computational cost.

- **Perturbation-based sharpness**  
  Small normalized random perturbations are added to the final parameters. The increase in validation loss is used as a simple estimate of how sensitive, or sharp, the final solution is.

## Main Results

In my experiments, **AdamW achieved lower mean validation loss**, while **Muon achieved slightly higher validation accuracy**.

Muon also produced noticeably larger relative parameter updates and required more training time.

Both optimizers continued reducing training loss over the three epochs, while validation loss started increasing after the first epoch. This suggests some overfitting under the chosen training budget.

The sharpness experiment did not give a strong enough signal to confidently claim that one optimizer found a universally flatter solution. I therefore treat the flatness result as a local and approximate measurement rather than a definitive conclusion.

## Running the Experiment

The complete experiment is contained in:

`diagnosing-optimization-during-fine-tuning.ipynb`

The notebook can be run on Kaggle or another environment with a GPU.

Install the required packages with:

```bash
pip install torch transformers datasets numpy pandas matplotlib
