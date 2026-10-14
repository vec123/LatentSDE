"""
Small networks and shape checks.

FCNN and multioutput_gradient: adapted from SVISE, reference/svise/svise/utils.py (GPL-3.0); the vmap branch
of multioutput_gradient, which needed a torch nightly build, is removed.
check_nn_dims: adapted from ROM-VISDE, reference/rom-visde/visde/utils.py (GPL-3.0).
"""
from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn

from .transforms import Identity


class FCNN(nn.Module):
    """
    Fully connected network.

    Args:
        input_size (int): input dimension
        hidden_size (int): units in each hidden layer
        output_size (int): output dimension
        num_layers (int): number of hidden layers
        nonlinearity (Callable): nonlinearity after each hidden layer
        output_transform (Callable): applied to the output
        batch_norm (bool): batch norm after each hidden layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        nonlinearity: Callable = nn.Softplus(),
        output_transform: Callable = Identity(),
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nonlinearity]
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nonlinearity]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
        layers += [nn.Linear(hidden_size, output_size), output_transform]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 0:
            return self.mlp(x.reshape(1, 1))
        if x.dim() == 1:
            return self.mlp(x.unsqueeze(-1))
        if x.dim() == 2:
            return self.mlp(x)
        bs = x.shape[:-1]
        return self.mlp(x.reshape(-1, x.shape[-1])).reshape(*bs, -1)


def multioutput_gradient(f: Tensor, t: Tensor) -> Tensor:
    """
    Gradient of the outputs f (bs, d) with respect to the times t (bs,) that produced them, row by row.

    Returns:
        Tensor: (bs, d) gradient
    """
    basis_vectors = torch.eye(f.shape[-1], dtype=f.dtype, device=f.device)
    rows = [torch.autograd.grad(f.sum(0), t, v, retain_graph=True)[0] for v in basis_vectors.unbind()]
    return torch.stack(rows, dim=-1)


def check_nn_dims(net, in_shapes, out_shapes, error_msg=None):
    """Raises if net does not map inputs of in_shapes to outputs of out_shapes (batch dimension excluded)."""
    if error_msg is None:
        error_msg = "check_nn_dims"
    n_batch = 42
    test_inputs = [torch.zeros(n_batch, *shape) for shape in in_shapes]
    try:
        test_outputs = net(*test_inputs)
    except Exception as e:
        raise ValueError(
            f"{error_msg}: Cannot evaluate nn Module with inputs of shape {[t.shape for t in test_inputs]}, "
            f"got exception {e}"
        )
    if not isinstance(test_outputs, tuple):
        test_outputs = (test_outputs,)
    assert len(out_shapes) == len(test_outputs), (
        f"{error_msg}: Expected {len(out_shapes)} outputs, got {len(test_outputs)}"
    )
    for i, shape in enumerate(out_shapes):
        expected = torch.Size([n_batch, *shape])
        assert test_outputs[i].shape == expected, (
            f"{error_msg}: Expected output {i} to have shape {expected}, got {test_outputs[i].shape}"
        )
