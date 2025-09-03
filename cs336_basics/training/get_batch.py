import numpy.typing as npt
import torch
from torch import Tensor
from jaxtyping import Int

def get_batch(
    dataset: npt.NDArray, # int array with token IDs,
    batch_size: int,
    context_size: int,
    device: str = "cpu", # or "mps" for Metal
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    """
    Load a random sample of data for training a language model.

    Returns:
        tuple[Tensor, Tensor]: Tuple of input and target arrays.
    """
    input_range = dataset.shape[0]
    max_start_idx = input_range - context_size - 1 # -1 so we can sample the corresponding y

    # High is exclusive here
    starts = torch.randint(0, max_start_idx + 1, size=(batch_size, ))

    data_tensor = torch.as_tensor(dataset, device=device)
    offsets = torch.arange(context_size, device=device)

    # Add an extra dimension to starts so we can broadcast the addition in 2 dimensions.
    idx = starts.unsqueeze(1) + offsets
    inputs = data_tensor[idx]
    targets = data_tensor[idx + 1]

    return (inputs, targets)