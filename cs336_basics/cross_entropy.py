import torch
from einops import einops
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # 1. Find the maximum logit for each example in the batch for stability.
    # The '1' in 'b 1' is equivalent to keepdim=True, which is essential for broadcasting.
    max_logits = einops.reduce(inputs, "b v -> b 1", "max")

    # 2. Compute the stable Log-Sum-Exp term.
    # Subtract the max to prevent overflow.
    stable_logits = inputs - max_logits
    # Sum the exponentiated stable logits over the vocabulary dimension.
    sum_exp_logits = einops.reduce(torch.exp(stable_logits), "b v -> b", "sum")
    # Take the log and add the max back.
    # We must squeeze max_logits from (b, 1) to (b,) for element-wise addition.
    log_sum_exp = torch.log(sum_exp_logits) + max_logits.squeeze(dim=-1)

    # 3. Gather the logits for the correct target classes.
    # A common and efficient PyTorch idiom for this is to use direct indexing.
    # We create a range for the batch dimension and use it with the targets tensor.
    batch_size = inputs.shape[0]
    batch_indices = torch.arange(batch_size, device=inputs.device)
    correct_class_logits = inputs[batch_indices, targets]

    # 4. Compute the loss for each example.
    # loss = log_sum_exp - logit_of_correct_class
    per_example_loss = log_sum_exp - correct_class_logits

    # 5. Return the average loss over the batch.
    # einops.reduce is a clear way to express this final reduction to a scalar.
    average_loss = einops.reduce(per_example_loss, "b -> ()", "mean")

    return average_loss