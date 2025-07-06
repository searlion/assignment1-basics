import torch
import numpy as np
import numpy.typing as npt


def data_loader(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # 1. Determine the number of possible starting positions for a sequence.
    # A sequence of length `context_length` and its target requires `context_length + 1` tokens.
    # The last possible starting index `i` must satisfy `i + context_length < len(dataset)`.
    # So, the valid starting indices are in the range [0, len(dataset) - context_length - 1].
    # `torch.randint`'s `high` parameter is exclusive, so the upper bound is len(dataset) - context_length.
    n = len(dataset)
    if n <= context_length:
        raise ValueError("Dataset is not long enough for the given context_length.")

    # 2. Sample `batch_size` random starting indices.
    # `ix` will be a tensor of shape (batch_size,) containing the starting indices.
    ix = torch.randint(0, n - context_length, (batch_size,))

    # 3. Create the input sequences (x) and target sequences (y) using list comprehensions.
    # For each random index `i` in `ix`, we slice the dataset to get a sequence.
    # `dataset[i:i+context_length]` is the input context.
    # `dataset[i+1:i+1+context_length]` is the target (the next token for each token in the input).
    # `torch.from_numpy` converts each slice into a tensor.
    # `torch.stack` combines the list of 1D tensors into a single 2D tensor.
    x = torch.stack([torch.from_numpy(dataset[i: i + context_length]) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1: i + 1 + context_length]) for i in ix])

    # 4. Move tensors to the specified device and ensure they are LongTensors.
    # Token IDs are categorical and used for lookups, so they should be integers (long).
    x, y = x.to(device), y.to(device)
    x, y = x.long(), y.long()

    return x, y