import pytest
import gym_electric_motor.utils as utils
import gym_electric_motor.core as core
import tests.testing_utils as dummies
import numpy as np


@pytest.mark.parametrize("state_names", [["a", "b", "c", "d"]])
@pytest.mark.parametrize(
    "state_dict, state_array, target",
    [
        # Set all Values Test and test for uppercase states
        (dict(A=5, b=12, c=10, d=12), [0, 0, 0, 0], [5, 12, 10, 12]),
        # Set only subset of values -> Rest stays as it is in the state array
        (dict(a=5, b=12, c=10), [0, 1, 2, 5], [5, 12, 10, 5]),
    ],
)
def test_state_dict_to_state_array(state_dict, state_array, state_names, target):
    utils.state_dict_to_state_array(state_dict, state_array, state_names)
    assert np.all(state_array == target)


def test_state_dict_to_state_array_assertion_error():
    # Test if the AssertionError is raised when invalid states are passed
    with pytest.raises(AssertionError):
        utils.state_dict_to_state_array(
            {"invalid_name": 0, "valid_name": 1}, np.zeros(3), ["valid_name", "a", "b"]
        )


@pytest.mark.parametrize("state_names", [["a", "b", "c", "d"]])
@pytest.mark.parametrize(
    "input_values, target",
    [
        (dict(a=5, b=12, c=10, d=12), [5, 12, 10, 12]),
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
        (5, [5, 5, 5, 5]),
    ],
)
def test_set_state_array(input_values, state_names, target):
    assert np.all(utils.set_state_array(input_values, state_names) == target)


@pytest.mark.parametrize("state_names", [["a", "b", "c", "d"]])
@pytest.mark.parametrize(
    "input_values, error",
    [
        ("a", Exception),
        (np.array([1, 2, 3]), AssertionError),
        ([1, 2, 3], AssertionError),
    ],
)
def test_set_state_array_exceptions(input_values, state_names, error):
    with pytest.raises(error):
        utils.set_state_array(input_values, state_names)
