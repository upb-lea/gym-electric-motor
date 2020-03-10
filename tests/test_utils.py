import pytest
import gym_electric_motor.utils as utils
import gym_electric_motor.core as core
import tests.testing_utils as dummies
import numpy as np


def mock_make_module(superclass, instance, **kwargs):
    return superclass, instance, kwargs


def test_registry(monkeypatch):
    registry = {}
    monkeypatch.setattr(utils, "_registry", registry)
    utils.register_superclass(core.PhysicalSystem)
    assert registry[core.PhysicalSystem] == {}

    utils.register_class(dummies.DummyPhysicalSystem, core.PhysicalSystem, 'DummySystem')
    assert registry[core.PhysicalSystem] == {'DummySystem': dummies.DummyPhysicalSystem}

    with pytest.raises(KeyError):
        _ = registry[dummies.DummyPhysicalSystem]

    with pytest.raises(KeyError):
        _ = registry[core.PhysicalSystem]['nonExistingKey']


def test_make_module(monkeypatch):
    registry = {}
    monkeypatch.setattr(utils, "_registry", registry)
    utils.register_superclass(core.PhysicalSystem)
    utils.register_class(dummies.DummyPhysicalSystem, core.PhysicalSystem, 'DummySystem')
    kwargs = dict(a=0, b=5)
    dummy_sys = utils.make_module(core.PhysicalSystem, 'DummySystem', **kwargs)
    assert isinstance(dummy_sys, dummies.DummyPhysicalSystem)
    assert dummy_sys.kwargs == kwargs

    with pytest.raises(Exception):
        _ = registry[core.ReferenceGenerator]['DummySystem']

    with pytest.raises(Exception):
        _ = registry[core.PhysicalSystem]['NonExistingKey']


def test_instantiate(monkeypatch):
    monkeypatch.setattr(utils, "make_module", mock_make_module)
    kwargs = dict(a=0, b=5)
    phys_sys = dummies.DummyPhysicalSystem(**kwargs)

    # Test object instantiation
    assert utils.instantiate(core.PhysicalSystem, phys_sys) == phys_sys

    # Test class instantiation
    instance = utils.instantiate(core.PhysicalSystem, dummies.DummyPhysicalSystem, **kwargs)
    assert type(instance) == dummies.DummyPhysicalSystem
    assert instance.kwargs == kwargs

    # Test string instantiation
    key = 'DummyKey'
    assert utils.instantiate(core.PhysicalSystem, key, **kwargs) == (core.PhysicalSystem, key, kwargs)

    # Test Exceptions
    with pytest.raises(Exception):
        utils.instantiate(core.ReferenceGenerator, dummies.DummyPhysicalSystem, **kwargs)
    with pytest.raises(Exception):
        utils.instantiate(dummies.DummyPhysicalSystem, core.ReferenceGenerator,  **kwargs)
    with pytest.raises(Exception):
        utils.instantiate(dummies.DummyPhysicalSystem(), core.ReferenceGenerator,  **kwargs)


@pytest.mark.parametrize("state_names", [['a', 'b', 'c', 'd']])
@pytest.mark.parametrize("state_dict, state_array, target", [
    # Set all Values Test and test for uppercase states
    (dict(A=5, b=12, c=10, d=12), [0, 0, 0, 0], [5, 12, 10, 12]),
    # Set only subset of values -> Rest stays as it is in the state array
    (dict(a=5, b=12, c=10), [0, 1, 2, 5], [5, 12, 10, 5]),
])
def test_state_dict_to_state_array(state_dict, state_array, state_names, target):
    utils.state_dict_to_state_array(state_dict, state_array, state_names)
    assert np.all(state_array == target)


def test_state_dict_to_state_array_assertion_error():
    # Test if the AssertionError is raised when invalid states are passed
    with pytest.raises(AssertionError):
        utils.state_dict_to_state_array({'invalid_name': 0, 'valid_name': 1}, np.zeros(3), ['valid_name', 'a', 'b'])


@pytest.mark.parametrize("state_names", [['a', 'b', 'c', 'd']])
@pytest.mark.parametrize("input_values, target", [
    (dict(a=5, b=12, c=10, d=12), [5, 12, 10, 12]),
    (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])),
    ([1, 2, 3, 4], [1, 2, 3, 4]),
    (5, [5, 5, 5, 5])
])
def test_set_state_array(input_values, state_names, target):
    assert np.all(utils.set_state_array(input_values, state_names) == target)


@pytest.mark.parametrize("state_names", [['a', 'b', 'c', 'd']])
@pytest.mark.parametrize("input_values, error", [
    ('a', Exception),
    (np.array([1, 2, 3]), AssertionError),
    ([1, 2, 3], AssertionError)
])
def test_set_state_array_exceptions(input_values, state_names, error):
    with pytest.raises(error):
        utils.set_state_array(input_values, state_names)
