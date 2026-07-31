
from benchmark.core.device import get_device
from benchmark.core.seed import set_seed


def test_seed():

    set_seed(42)


def test_device():

    device = get_device()

    assert str(device) in (
        "cpu",
        "cuda",
    )
