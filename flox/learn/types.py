import torch
import torch
import typing as t


Kind: t.TypeAlias = t.Literal["async", "sync", "sync-v2"]
"""..."""

Where: t.TypeAlias = t.Literal["local", "globus_compute"]
"""..."""

Params: t.TypeAlias = dict[str, torch.Tensor]  # torch.optim.Optimizer.StateDict
"""The state dict of PyTorch ``torch.learn.Module`` (see ``torch.learn.Module.params()``)."""

Loss: t.TypeAlias = torch.Tensor
"""..."""
