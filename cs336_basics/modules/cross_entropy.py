import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy_loss(predicted: Float[Tensor, "batch_size vocab"],
                       target: Int[Tensor, "batch_size"]) -> Float[Tensor, ""]:
    """
    Cross-entropy loss is negative log likelihood:

    loss(theta, D) = (avg across training set) - 1/m * Sum(i=1 to m) log p_theta(x_i+1 | x_i)
    where p_theta is the PDF for the model.

    Actual computed logits p(x_i+1 | x_i) is exp(o_i[x_i+1]) / Sum_j exp(o_i[j])

    So log(p) = o_i[x_i+1] - log(Sum_j exp(o_i[j])) = o_i[x_i+1] - o_i[j]_max - log(Sum_j exp(o_i[j] - o_i[j]_max))
    """
    # We need to return the average per batch.
    batch_size = target.shape[0]

    indices_for_gather = target.view((batch_size, 1))
    correct_logits = torch.gather(predicted, dim=-1, index=indices_for_gather).squeeze(dim=-1)

    max_predicted = torch.max(predicted, dim=-1).values.unsqueeze(-1)
    denominator = (predicted - max_predicted)
    output = correct_logits - max_predicted - denominator.exp().sum(dim=-1).log()

    return -output.mean()