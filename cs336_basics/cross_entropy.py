import torch
from einops import einops
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(inputs: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "... vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "..."]): Tensor of shape (...) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 1. Find the maximum logit for each example for stability.
    max_logits = einops.reduce(inputs, "... v -> ... 1", "max")

    # 2. Compute the stable Log-Sum-Exp term.
    stable_logits = inputs - max_logits
    log_sum_exp = torch.log(einops.reduce(torch.exp(stable_logits), "... v -> ...", "sum"))
    log_sum_exp += max_logits.squeeze(dim=-1)

    # 3. Gather the logits for the correct target classes.
    correct_class_logits = torch.gather(inputs, -1, targets.unsqueeze(-1)).squeeze(-1)

    # 4. Compute the loss for each example.
    per_example_loss = log_sum_exp - correct_class_logits

    # 5. Return the average loss.
    return einops.reduce(per_example_loss, "... -> ()", "mean")