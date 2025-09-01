import torch
import os
import typing

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    dest: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    output = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
    }
    torch.save(output, dest)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    # Returns the iteration
    checkpoint_data = torch.load(src)
    model.load_state_dict(checkpoint_data["model"])
    optimizer.load_state_dict(checkpoint_data["optim"])
    return checkpoint_data["iteration"]