import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.generation import decode
import json
import argparse

def main():

    parser = argparse.ArgumentParser(description="Train a TransformerLM model.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length.")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, default=1024, help="Feedforward dimension.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta.")

    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file.")
    parser.add_argument("--merges_path", type=str, required=True, help="Path to merges file.")
    args = parser.parse_args()

    # Load the tokenizer
    vocab_filepath = "/home/lewis/github/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    merges_filepath = "/home/lewis/github/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    
    with open(vocab_filepath, 'r') as f:
        gpt2_vocab = json.load(f)

    with open(merges_filepath, 'r', encoding="utf-8") as f:
        gpt2_merges_lines = f.read().split('\n')

    # Convert vocab from json to the required format (Dict[int, bytes])
    vocab = {i: s.encode('utf-8') for s, i in gpt2_vocab.items()}

    # Convert merges from list of strings to the required format (List[Tuple[bytes, bytes]])
    merges = []
    for merge_line in gpt2_merges_lines:
        if merge_line.startswith('#') or not merge_line:
            continue
        p1, p2 = merge_line.split()
        merges.append((p1.encode('utf-8'), p2.encode('utf-8')))

    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
    )

    # Load the model
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLM(**model_config)
    checkpoint = torch.load("/home/lewis/github/assignment1-basics/checkpoints/model_iter_900.pt")
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model.eval()

    # Generate text
    prompt = "Once upon a time, there was a pretty girl named Lily"
    generated_text = decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=256,
        temperature=0.9,
        top_p=0.85,
    )

    print(generated_text)

if __name__ == "__main__":
    main()
