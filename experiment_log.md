# Experiment Log

This log documents the experiments run for the CS336 assignment.

### Tokenizer

First, run the following script to tokenise the text:

```bash
python -m scripts.train_bpe_and_tokenize \
--input_file data/TinyStoriesV2-GPT4-train.txt \
--vocab_size 10000 \
--output_dir tokenizers/tinystories_10k_train
```

### Run experiment

```bash
python -m scripts.train \
   --train_data_path tokenizers/tinystories_10k_train/tokens.bin \
   --valid_data_path tokenizers/tinystories_10k_valid/tokens.bin \
   --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
   --merges_path tokenizers/tinystories_10k_train/merges.pkl \
   --d_model 256 \
   --num_layers 4 \
   --num_heads 4 \
   --batch_size 64 \
   --max_iters 1000
```

### Generate

```bash
python generate_text.py \
   --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
   --merges_path tokenizers/tinystories_10k_train/merges.pkl \
   --d_model 256 \
   --num_layers 4 \
   --num_heads 4 
```

### Using main.py
ith custom parameters
```bash
uv run main.py \
    --input_file data/TinyStoriesV2-GPT4-train.txt \
    --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
    --merges_path tokenizers/tinystories_10k_train/merges.pkl \
    --train_tokens_file data/train_tokens.npy \
    --valid_tokens_file data/valid_tokens.npy \
    --max_iters 2000 \
    --batch_size 128 \
    --lr 2e-4
```
Skip tokenization if already done

```bash
uv run main.py \
    --input_file data/TinyStoriesV2-GPT4-train.txt \
    --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
    --merges_path tokenizers/tinystories_10k_train/merges.pkl \
    --skip_tokenization
```

## Experiment 1: Baseline Model

*   **Description**: Train the baseline TransformerLM model with the default hyperparameters.
*   **Command**:

```bash
python -m scripts.train_bpe_and_tokenize \
  --input_file data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --output_dir tokenizers/tinystories_10k_train
```
    
```bash
python cs336_basics/train.py \
   --train_data_path tokenizers/tinystories_10k_train/tokens.bin \
   --valid_data_path tokenizers/tinystories_10k_valid/tokens.bin \
   --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
   --merges_path tokenizers/tinystories_10k_train/merges.pkl \
   --d_model 256 \
   --num_layers 4 \
   --num_heads 4 \
   --batch_size 64 \
   --max_iters 1000
```

*   **Observations**: 

Based on the provided graphs, there is a very clear and significant issue with this machine learning model's training process: **severe overfitting**.

Here is a breakdown of what the graphs show and why it's a problem:

### The Main Problem: Overfitting

The most telling evidence comes from comparing the **`val_loss` (validation loss)** and **`train_loss` (training loss)** graphs.

1.  **`train_loss` (Good Behavior):** This graph shows the model's error on the data it's being trained on. As expected, the loss consistently decreases over time. This means the model is successfully learning to minimize its error on the training examples.

2.  **`val_loss` (Problematic Behavior):** This graph shows the model's error on a separate "validation" dataset that it does not see during training. This is the true measure of how well the model can generalize to new, unseen data.
    *   The validation loss starts high, drops to a minimum around **Step 3**, and then **steadily increases** for the rest of the training.

**This divergence is the classic sign of overfitting.**

*   **What it means:** After Step 3, the model is no longer learning general patterns from the data. Instead, it has started to "memorize" the specific examples and noise in the training set. This is why its performance on the training data keeps getting better (`train_loss` goes down), but its performance on new data gets worse (`val_loss` goes up).

### Analysis of Other Graphs

*   **`lr` (Learning Rate):** This graph shows a "warm-up and decay" learning rate schedule. The learning rate starts low, increases to a peak (around 0.001), and then gradually decreases. This is a standard and often effective technique. While the schedule itself isn't "wrong," the chosen peak value or the length of training might be contributing to the overfitting.

*   **`iteration`:** This graph simply confirms that the training process is running and the number of iterations is increasing with each step. It doesn't indicate a problem on its own.

### Summary and Recommendations

**In short, yes, there is something wrong.** The model is overfitting badly. The best version of this model was produced around **Step 3**, where the validation loss was at its lowest. Continuing to train beyond that point made the model progressively worse at its actual task.

**How to fix this:**

1.  **Early Stopping:** This is the most important fix. You should monitor the validation loss and stop the training process (or save the model checkpoint) when the `val_loss` stops decreasing and starts to rise. Based on this chart, you should have stopped at Step 3.

2.  **Regularization:** Introduce or increase regularization techniques to prevent the model from memorizing the training data. Common methods include:
    *   **Dropout:** Randomly deactivating neurons during training.
    *   **Weight Decay (L2 Regularization):** Penalizing large weights in the model.

3.  **Data Augmentation:** Create more training data by applying small, realistic transformations to your existing data. This makes it harder for the model to overfit and encourages it to learn more robust features.

4.  **Reduce Model Complexity:** If the model is too large or complex for the amount of data you have, it can easily overfit. You could try using a smaller version of the LLM.

5.  **Adjust the Learning Rate:** A lower peak learning rate might lead to slower but more stable convergence, potentially finding a better minimum for the validation loss.

*   **W&B Link**: https://wandb.ai/lewis-won-sear/cs336_assignment1/runs/xoah4sj6/workspace?nw=nwuserlewiswon

---

## Experiment 2: ...

*   **Description**: ...
*   **Command**: ...
*   **Observations**: ...
*   **W&B Link**: ...


