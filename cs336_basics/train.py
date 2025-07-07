
import argparse
import os
import time
import multiprocessing
from tqdm import tqdm

import numpy as np
import torch
import wandb

from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loader import data_loader
from cs336_basics.learning_rate_scheduler import learning_rate_scheduler
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM


def main():
    parser = argparse.ArgumentParser(description="Train a TransformerLM model.")

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length.")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feedforward dimension.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta.")

    # Optimizer hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Maximum learning rate.")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="Minimum learning rate.")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="AdamW betas.")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Number of warmup iterations.")
    parser.add_argument("--cosine_cycle_iters", type=int, default=1000, help="Number of cosine cycle iterations.")

    # Data and training parameters
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data.")
    parser.add_argument("--valid_data_path", type=str, required=True, help="Path to validation data.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file.")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum training iterations.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")

    # W&B logging
    parser.add_argument("--wandb_project", type=str, default="cs336_assignment1", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="transformer_lm_training", help="W&B run name.")

    args = parser.parse_args()

    # Initialize W&B
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
    )

    # Load data
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')
    
    # Model initialization
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLM.from_config(model_config).to(args.device)

    # Optimizer initialization
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Training loop
    start_time = time.time()
    for i in range(args.max_iters):
        # Learning rate decay
        lr = learning_rate_scheduler(
            it=i,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get a batch of data
        x, y = data_loader(train_data, args.batch_size, args.context_length, args.device)

        # Forward pass
        logits = model(x)
        loss = cross_entropy(logits, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if i % args.eval_interval == 0:
            end_time = time.time()
            print(f"Iteration {i}: Train Loss = {loss.item():.4f}, LR = {lr:.6f}, Time = {end_time - start_time:.2f}s")
            wandb.log({"train_loss": loss.item(), "lr": lr, "iteration": i})
            start_time = time.time()

            # Validation
            model.eval()
            with torch.no_grad():
                val_x, val_y = data_loader(valid_data, args.batch_size, args.context_length, args.device)
                val_logits = model(val_x)
                val_loss = cross_entropy(val_logits, val_y)
                print(f"Iteration {i}: Valid Loss = {val_loss.item():.4f}")
                wandb.log({"val_loss": val_loss.item(), "iteration": i})
            model.train()

            # Checkpointing
            checkpoint_path = os.path.join(args.checkpoint_dir, f"model_iter_{i}.pt")
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model.to_config(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
