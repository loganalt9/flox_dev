from abc import ABC, abstractmethod
from concurrent.futures import Future

from flox.flock import FlockNode
from flox.jobs import Job, NodeCallable


# @dataclass
# class LauncherConfig:
#     kind: LauncherKind
#     args: LauncherArgs


class Launcher(ABC):
    """
    Base class for launching functions in an FL process.
    """

    def __init__(self):
        pass

    @abstractmethod
    def submit(self, fn: NodeCallable, node: FlockNode, /, *args, **kwargs) -> Future:
        raise NotImplementedError()

    @abstractmethod
    def collect(self):
        # TODO: Check if this is needed at all.
        raise NotImplementedError()
