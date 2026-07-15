from state_machine import BoundedAccumulator


def test_accumulator_saturates_upper_bound() -> None:
    accumulator = BoundedAccumulator(0.0, 10.0, 8.0)
    assert accumulator.apply(5.0) == 10.0
