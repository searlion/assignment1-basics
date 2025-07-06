import torch
from typing import Iterable

# This is the adapter function you need to fill in.
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # Epsilon for numeric stability, as specified in the problem
    epsilon = 1e-6

    # 1. Collect all non-None gradients from the parameters.
    # We only consider parameters that have gradients.
    grads = [p.grad for p in parameters if p.grad is not None]

    # If there are no gradients, there's nothing to do.
    if len(grads) == 0:
        return

    # 2. Compute the total l2-norm of all gradients combined.
    # A memory-efficient way is to sum the squared norms of each grad tensor
    # and then take the square root of the total sum.
    total_norm = torch.sqrt(torch.sum(torch.stack([torch.sum(g.pow(2)) for g in grads])))

    # 3. Check if clipping is needed. If the norm is already within the limit, do nothing.
    if total_norm <= max_l2_norm:
        return

    # 4. Calculate the clipping coefficient (scaling factor).
    # scale = M / (||g|| + epsilon)
    clip_coef = max_l2_norm / (total_norm + epsilon)

    # 5. Scale down all gradients in-place.
    for g in grads:
        g.mul_(clip_coef)