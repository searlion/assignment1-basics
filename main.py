import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tokenization and training pipeline.")

    # Data paths
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input text file for tokenization.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to the vocabulary file.")
    parser.add_argument("--merges_path", type=str, required=True,
                        help="Path to the merges file.")

    # Output paths
    parser.add_argument("--train_tokens_file", type=str, default="train_tokens.npy",
                        help="Path to save tokenized training data.")
    parser.add_argument("--valid_tokens_file", type=str, default="valid_tokens.npy",
                        help="Path to save tokenized validation data.")

    # Training parameters (with defaults)
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1344, help="Feedforward dimension.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum training iterations.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory.")

    # Optional: skip tokenization if tokens already exist
    parser.add_argument("--skip_tokenization", action="store_true",
                        help="Skip tokenization if token files already exist.")

    args = parser.parse_args()

    # Check if required files exist
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        sys.exit(1)

    if not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file {args.vocab_path} not found.")
        sys.exit(1)

    if not os.path.exists(args.merges_path):
        print(f"Error: Merges file {args.merges_path} not found.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.train_tokens_file) if os.path.dirname(args.train_tokens_file) else ".",
                exist_ok=True)

    # Step 1: Tokenize the dataset
    if not args.skip_tokenization or not (
            os.path.exists(args.train_tokens_file) and os.path.exists(args.valid_tokens_file)):
        print("Step 1: Tokenizing dataset...")

        # Tokenize training data
        tokenize_cmd = [
            "python", "tokenize_dataset.py",
            "--input_file", args.input_file,
            "--output_file", args.train_tokens_file,
            "--vocab_path", args.vocab_path,
            "--merges_path", args.merges_path
        ]

        if not run_command(tokenize_cmd, "Tokenizing training data"):
            print("Failed to tokenize training data. Exiting.")
            sys.exit(1)

        # For now, we'll use the same tokenized data for validation
        # In a real scenario, you'd have separate train/valid text files
        print("Using the same tokenized data for validation (you may want to split your data beforehand)")
        if args.train_tokens_file != args.valid_tokens_file:
            import shutil
            shutil.copy(args.train_tokens_file, args.valid_tokens_file)
    else:
        print("Skipping tokenization - token files already exist.")

    # Step 2: Train the model
    print("\nStep 2: Training the model...")

    train_cmd = [
        "python", "train.py",
        "--train_data_path", args.train_tokens_file,
        "--valid_data_path", args.valid_tokens_file,
        "--vocab_path", args.vocab_path,
        "--merges_path", args.merges_path,
        "--vocab_size", str(args.vocab_size),
        "--context_length", str(args.context_length),
        "--d_model", str(args.d_model),
        "--num_layers", str(args.num_layers),
        "--num_heads", str(args.num_heads),
        "--d_ff", str(args.d_ff),
        "--batch_size", str(args.batch_size),
        "--max_iters", str(args.max_iters),
        "--lr", str(args.lr),
        "--checkpoint_dir", args.checkpoint_dir
    ]

    if not run_command(train_cmd, "Training the model"):
        print("Failed to train the model. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Tokenized data saved to: {args.train_tokens_file}, {args.valid_tokens_file}")
    print(f"Model checkpoints saved to: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()