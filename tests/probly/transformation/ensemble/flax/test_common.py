import pytest
from probly.transformation.ensemble.common import ensemble, register


class DummyPredictor:
    pass


def test_register_then_ensemble_calls_generator_and_returns_value():
    calls = []

    def fake_generator(base, n_members, reset_params=True):
        calls.append((base, n_members, reset_params))
        return "OK"

    register(DummyPredictor, fake_generator)

    base = DummyPredictor()
    result = ensemble(base, 5)  # nur base und num_members

    assert result == "OK"
    assert calls == [(base, 5, True)]  # reset_params defaultet auf True





def test_ensemble_raises_if_type_not_registered():
    class UnregisteredPredictor:
        pass

    base = UnregisteredPredictor()

    with pytest.raises((TypeError, NotImplementedError)) as e:
        ensemble(base, 3)

    msg = str(e.value)
    assert (
        "unexpected keyword argument 'n_members'" in msg
        or "No ensemble generator is registered" in msg
    )
