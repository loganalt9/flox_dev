import typing as t

import lightning as pl
import numpy as np
from torch.utils.data import Dataset, Subset

from flight.federation.topologies.node import Node, NodeID

# Flight, SKLearn, PyTorch Lightning, PyTorch

# simulated: inidices

CustomDataset: t.TypeAlias = t.Any

class DataLoadable(t.Protocol):
    data: Dataset | np.ndarray
    indices: dict | None

    def __iter__(self) -> t.Iterator[t.Any]:
        pass

    def __next__(self) -> t.Any:
        pass

    def load(self, node: Node) -> Subset | np.ndarray | CustomDataset:
        pass
    

class TorchLoadable:
    def __init__(
        self, data: Dataset, indices: dict[NodeID, t.Sequence[int]] | None = None
    ) -> None:
        self.data = data
        self.indices = indices
    
    def __iter__(self) -> t.Iterator[t.Any]:
        self.it = 0
        return iter(self.data)
    
    def __next__(self) -> t.Any:
        step = self.data[self.it]
        self.it += 1
        return step

    def load(self, node: Node) -> Subset | np.ndarray | CustomDataset:
        if self.indices:
            return Subset(self.data, self.indices[node.idx])
        return self.data


class SKLoadable:
    def __init__(
        self, data: np.ndarray, indices: dict[NodeID, t.Sequence[int]]
    ) -> None:
        self.data = data
        self.indices = indices

    def __iter__(self) -> t.Iterator[t.Any]:
        self.it = 0
        return iter(self.data)
    
    def __next__(self) -> t.Any:
        step = self.data[self.it]
        self.it += 1
        return step

    def load(self, node: Node) -> Subset | np.ndarray | CustomDataset:
        if self.indices:
            indices = self.indices[node.idx]
            subset = self.data[indices[0] : indices[len(indices) - 1]]
            return subset
        return self.data


class LightningLoadable:
    def __init__(
        self, data: Dataset, indices: dict[NodeID, t.Sequence[int]] | None = None
    ) -> None:
        self.data = data
        self.indices = indices

    def load(self, node: Node) -> Subset | np.ndarray | CustomDataset:
        if self.indices:
            return Subset(self.data, self.indices[node.idx])
        return self.data
    
