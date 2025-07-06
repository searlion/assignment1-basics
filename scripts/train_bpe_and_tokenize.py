
import argparse
import os
import pickle
import numpy as np
import multiprocessing
from tqdm import tqdm

from cs336_basics.train_bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

_tokenizer = None

def init_tokenizer(vocab_path, merges_path, special_tokens):
    global _tokenizer
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    _tokenizer = Tokenizer(vocab, merges, special_tokens)

def count_tokens_in_chunk(chunk):
    return len(_tokenizer.encode(chunk))

def write_tokens_in_chunk(chunk):
    return _tokenizer.encode(chunk)

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and tokenize a dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenizer and tokenized data.")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens.")
    parser.add_argument("--chunk_size", type=int, default=1024 * 1024, help="Size of chunks to read from the input file.")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Train the tokenizer
    print("Training BPE tokenizer...")
    training_result = train_bpe(
        input_path=args.input_file,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    # Save the tokenizer
    vocab_path = os.path.join(args.output_dir, "vocab.pkl")
    merges_path = os.path.join(args.output_dir, "merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(training_result.vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(training_result.merges, f)

    print(f"Tokenizer saved to {args.output_dir}")

    # Tokenize the dataset
    print("Tokenizing dataset...")

    # Set up the tokenizer for parallel processing
    init_tokenizer(vocab_path, merges_path, args.special_tokens)

    # Get file size for tqdm
    file_size = os.path.getsize(args.input_file)

    # Count tokens in parallel
    print("Counting tokens...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        with multiprocessing.Pool(initializer=init_tokenizer, initargs=(vocab_path, merges_path, args.special_tokens)) as pool:
            chunks = iter(lambda: f.read(args.chunk_size), "")
            num_tokens = sum(tqdm(pool.imap(count_tokens_in_chunk, chunks), total=file_size // args.chunk_size, desc="Counting tokens"))

    output_file = os.path.join(args.output_dir, "tokens.bin")
    print(f"Saving {num_tokens} tokens to {output_file}...")
    fp = np.memmap(output_file, dtype=np.uint16, mode='w+', shape=(num_tokens,))

    # Write tokens in parallel
    print("Writing tokens...")
    offset = 0
    with open(args.input_file, 'r', encoding='utf-8') as f:
        with multiprocessing.Pool(initializer=init_tokenizer, initargs=(vocab_path, merges_path, args.special_tokens)) as pool:
            chunks = iter(lambda: f.read(args.chunk_size), "")
            for tokens in tqdm(pool.imap(write_tokens_in_chunk, chunks), total=file_size // args.chunk_size, desc="Writing tokens"):
                fp[offset : offset + len(tokens)] = tokens
                offset += len(tokens)

    fp.flush()
    print("Done.")

if __name__ == "__main__":
    main()
