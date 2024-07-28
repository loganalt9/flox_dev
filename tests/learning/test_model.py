import pytest
import torch

from flight.learning.mytorch import FlightModule


@pytest.fixture
def valid_module():
    class TestModule(FlightModule):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(1, 10),
                torch.nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_nb):
            return self(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    return TestModule


@pytest.fixture
def invalid_module():
    class TestModule(FlightModule):  # noqa
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(1, 10),
                torch.nn.Linear(10, 1),
            )

        def forward(self, x):
            return self.model(x)

    return TestModule


class TestModelInit:
    def test_1(self, valid_module):
        model = valid_module()
        assert isinstance(model, FlightModule)
        assert isinstance(model, torch.nn.Module)

        x = torch.tensor([[1.0]])
        y = model(x)
        assert isinstance(y, torch.Tensor)

    def test_2(self, invalid_module):
        with pytest.raises(TypeError):
            invalid_module()
