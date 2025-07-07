
import torch
from torch.nn import functional as F

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def decode(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """ 
    Generates text from a model given a prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The input text to condition the generation on.
        max_new_tokens: The maximum number of tokens to generate.
        temperature: The temperature for sampling.
        top_p: The nucleus sampling probability.

    Returns:
        The generated text.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode the prompt
    prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # Generate tokens
    for _ in range(max_new_tokens):
        # Get the last `context_length` tokens
        if prompt_tokens.size(1) > model.to_config()["context_length"]:
            context_tokens = prompt_tokens[:, -model.to_config()["context_length"]:]
        else:
            context_tokens = prompt_tokens

        # Get the model logits
        logits = model(context_tokens)
        logits = logits[:, -1, :]  # Get the logits for the last token

        # Apply temperature scaling
        logits = logits / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float("Inf")

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append the new token to the sequence
        prompt_tokens = torch.cat([prompt_tokens, next_token], dim=1)

        # Check for end-of-text token
        if next_token.item() == tokenizer.encoder.get(b"<|endoftext|>"):
            break

    # Decode the generated tokens
    return tokenizer.decode(prompt_tokens[0].tolist())
