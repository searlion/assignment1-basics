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
python cs336_basics/train.py \
   --train_data_path data/TinyStoriesV2-GPT4-train.txt \
   --valid_data_path data/TinyStoriesV2-GPT4-valid.txt \
   --vocab_path tokenizers/tinystories_10k_train/vocab.pkl \
   --merges_path tokenizers/tinystories_10k_train/merges.pkl \
   --d_model 256 \
   --num_layers 4 \
   --num_heads 4 \
   --batch_size 64 \
   --max_iters 1000
```

## Experiment 1: Baseline Model

*   **Description**: Train the baseline TransformerLM model with the default hyperparameters.
*   **Command**:
    ```bash
    python -m cs336_basics.train \
        --train_data_path data/owt_train.txt \
        --valid_data_path data/owt_valid.txt \
        --vocab_path tests/fixtures/gpt2_vocab.json \
        --merges_path tests/fixtures/gpt2_merges.txt \
        --checkpoint_dir checkpoints \
        --wandb_run_name "baseline"
    ```
*   **Observations**: [Record your observations here. e.g., final validation loss, training speed, any anomalies.]
*   **W&B Link**: [Link to your W&B run]

---

## Experiment 2: ...

*   **Description**: ...
*   **Command**: ...
*   **Observations**: ...
*   **W&B Link**: ...


