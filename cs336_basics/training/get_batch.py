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

    # TODO: Is there a better way to do this?
    data_tensor = torch.tensor(dataset, device=device)
    unfolded_tensor = data_tensor.unfold(0, context_size, 1)
    inputs = unfolded_tensor[starts]
    targets = unfolded_tensor[starts + 1]

    return (inputs, targets)