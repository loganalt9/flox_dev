from torchvision.datasets import MNIST
import numpy as np

from flight.federation.topologies.node import NodeKind
from flight.learning.datasets import DataLoadable, TorchLoadable, SKLoadable
from flight.federation.topologies.io import from_dict

class test_torchloadable:
    def test_sim():
        topo = {
            0: {
                "kind": "coordinator",
                "globus_comp_id": None,
                "proxystore_id": None,
                "extra": None,
                "children": [1, 2],
            },
            1: {
                "kind": "aggregator",
                "globus_comp_id": None,
                "proxystore_id": None,
                "extra": None,
                "children": [3, 4],
            },
            2: {
                "kind": "aggregator",
                "globus_comp_id": None,
                "proxystore_id": None,
                "extra": None,
                "children": [5, 6],
            },
        }
        for idx in (3, 4, 5, 6):
            topo[idx] = {
                "kind": "worker",
                "globus_comp_id": None,
                "proxystore_id": None,
                "extra": None,
                "chilren": [],
            }

        nodes, links = from_dict(topo)

        data = MNIST("../torch_datasets", download=True)
        indices = {
            3: range(0, 1000),
            4: range(1000, 2000),
            5: range(2000, 3000),
            6: range(3000, 4000),
        }
        dl = TorchLoadable(data, indices)

        node_data = {}
        for node in nodes:
            if node.kind == NodeKind.WORKER:
                node_data[node.idx] = dl.load(node)

        for data in node_data.values():
            assert len(data) == 1000

    def test_iteration():
        data = MNIST("../torch_datasets", download=True)
        
        dl = TorchLoadable(data)

        for d in dl:
            assert isinstance(d, tuple)

class test_sklearnloadable:
    def test_iteration():
        arr = []
        data = np.array()

        dl = SKLoadable()