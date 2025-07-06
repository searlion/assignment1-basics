
import argparse
import numpy as np
from cs336_basics.tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Tokenize a dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output numpy memmap file.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocabulary file.")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to the merges file.")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
    )

    print(f"Tokenizing {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokens = tokenizer.encode(text)
    
    print(f"Saving tokenized data to {args.output_file}...")
    fp = np.memmap(args.output_file, dtype=np.uint16, mode='w+', shape=(len(tokens),))
    fp[:] = tokens[:]
    fp.flush()
    print("Done.")

if __name__ == "__main__":
    main()
