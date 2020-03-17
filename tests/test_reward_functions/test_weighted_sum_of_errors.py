import pytest
import numpy as np

from gym_electric_motor.reward_functions.weighted_sum_of_errors import WeightedSumOfErrors, ShiftedWeightedSumOfErrors
import gym_electric_motor.reward_functions.weighted_sum_of_errors as ws
import tests.testing_utils as tu


class TestWeightedSumOfErrors:
    """
    This test class consists of unit tests to verify modules in the weighted_sum_of_errors.py
    Following modules are included:
    WeightedSumOfErrors
        __init__
        set_modules

    ShiftedWeightedSumOfErrors
        set_modules
    """

    # Parameters used for testing
    reward_dict = dict(omega=1, torque=1, i_a=1)
    reward_list = [1, 1, 1]
    # Test counter to count function calls
    _mock_set_state_array_counter = 0

    @pytest.mark.parametrize("reward_weights", [None, reward_dict, reward_list])
    @pytest.mark.parametrize("normed", [False, True])
    @pytest.mark.parametrize("power", [-1, 1, 0])
    def test_initialization(self, reward_weights, normed, power):
        """
        Tests the initialization of WeightedSumOfErrors object with defined parameters
        :param reward_weights:
        :param normed:
        :param power:
        :return:
        """
        test_wsoe_obj = WeightedSumOfErrors(reward_weights, normed, observed_states=None, gamma=0.9, reward_power=power)
        # test if the correct attributes are set during init
        assert test_wsoe_obj._n == power
        assert test_wsoe_obj._reward_weights == reward_weights
        assert not test_wsoe_obj._state_length
        assert test_wsoe_obj._normed == normed

    @pytest.mark.parametrize("reward_weights", [None, reward_list, reward_dict])
    @pytest.mark.parametrize("normed", [True, False])
    @pytest.mark.parametrize("power", [-1, 1, 2, 0])
    @pytest.mark.parametrize("mock_ref_states", [[True, True], [True, False]])
    def test_wsoe_set_modules(self, monkeypatch, reward_weights, normed, power, mock_ref_states):
        """
         test the set_modules function
        :param monkeypatch:
        :param reward_weights:
        :param normed:
        :param power:
        :param mock_ref_states:
        :return:
        """
        expected_reward_range = (-3, 0)
        expected_reward_range_normed = (-1, 0)
        expected_reward = 1/3
        expected_reward_normed = 1
        expected_call_counter = 2

        def mock_set_state_array(a, b):
            """
            mock function for set_state_array() function
            :param a:
            :param b:
            :return: a fixed array [1, 1, 1] if parameter a is an array else a fixed value 2
            """
            self._mock_set_state_array_counter += 1
            test_b = np.array(["dummy_state_0", "dummy_state_1"])

            # Verify if the correct parameters are passed to the function
            for i in range(2):
                assert b[i] == test_b[i]
            if type(a) is list or np.ndarray:
                return np.array([1, 1, 1], dtype=np.float64)
            else:
                return 2

        wsoe_obj = WeightedSumOfErrors(reward_weights, normed, observed_states=None, gamma=0.9, reward_power=power)
        # create a dummy physical system with 2 states
        physical_system = tu.DummyPhysicalSystem(2)
        # create a dummy reference generator
        reference_generator = tu.DummyReferenceGenerator()
        monkeypatch.setattr(ws, "set_state_array", mock_set_state_array)
        # mocking the reference_generator._referenced_states attribute
        reference_generator._referenced_states = mock_ref_states
        wsoe_obj.set_modules(physical_system, reference_generator)

        # Test if the mock_set_state_array function is called only 2 times
        assert self._mock_set_state_array_counter == expected_call_counter, \
            'Function is not called as often as expected'
        # Test the reward weights and range
        if normed:
            assert wsoe_obj.reward_range == expected_reward_range_normed  # (-1, 0)
            for val in wsoe_obj._reward_weights:
                assert val == expected_reward   # 1 / 3

        else:
            assert wsoe_obj.reward_range == expected_reward_range  # (-3, 0)
            for val in wsoe_obj._reward_weights:
                assert val == expected_reward_normed  # 1

    @pytest.mark.parametrize("reward_weights", [None, reward_list, reward_dict])
    @pytest.mark.parametrize("normed", [True, False])
    @pytest.mark.parametrize("power", [-1, 0, 1, 2])
    @pytest.mark.parametrize("mock_ref_states", [[True, True], [True, False]])
    def test_wsoe_set_modules_warning(self, monkeypatch, reward_weights, normed, power, mock_ref_states):
        """
        test the warning raised in the set_module function
        :param monkeypatch:
        :param reward_weights:
        :param normed:
        :param power:
        :param mock_ref_states:
        :return:
        """
        def mock_set_state_array(a, _):
            """
             mock function for set_state_array() function
            :param a:
            :param _:
            :return: a fixed array [0, 0, 0] if parameter a so that a warning is raised

            """
            self._mock_set_state_array_counter += 1
            if type(a) is list or np.ndarray:
                return np.array([0, 0, 0], dtype=np.float64)
            else:
                return 2

        wsoe_obj = WeightedSumOfErrors(reward_weights, normed, observed_states=None, gamma=0.9, reward_power=power)
        physical_system = tu.DummyPhysicalSystem(2)  # setup_physical_system(motor_type, converter_type)
        reference_generator = tu.DummyReferenceGenerator()
        monkeypatch.setattr(ws, "set_state_array", mock_set_state_array)
        reference_generator._referenced_states = mock_ref_states  # parametrized to [true/false]

        # assert, if the correct warning is raised
        with pytest.warns(Warning, match="All reward weights sum up to zero"):
            wsoe_obj.set_modules(physical_system, reference_generator)

    @pytest.mark.parametrize("reward_weights", [None, reward_list, reward_dict])
    @pytest.mark.parametrize("normed", [True, False])
    @pytest.mark.parametrize("power", [-1, 1, 2, 0])
    @pytest.mark.parametrize("mock_ref_states", [[True, True], [True, False]])
    def test_shifted_wsoe_set_modules(self, monkeypatch, reward_weights, normed, power, mock_ref_states):
        """
        test the set_modules function in the ShiftedWeightedSumOfErrors class
        :param monkeypatch:
        :param reward_weights:
        :param normed:
        :param power:
        :param mock_ref_states:
        :return:
        """
        expected_reward_range = (0, 3)
        expected_reward_range_normed = (0, 1)
        expected_reward = 1/3
        expected_reward_normed = 1
        expected_call_counter = 2

        def mock_set_state_array(a, b):
            """
            mock function for set_state_array() function
            :param a:
            :param b:
            :return: a fixed array [1, 1, 1] if parameter a is an array else a fixed value 2
            """
            self._mock_set_state_array_counter += 1
            test_b = np.array(["dummy_state_0", "dummy_state_1"])
            for i in range(2):
                assert b[i] == test_b[i]
            if type(a) is list or np.ndarray:
                return np.array([1, 1, 1], dtype=np.float64)
            else:
                return 2

        # create the ShiftedWeightedSumOfErrors object
        swsoe_obj = ShiftedWeightedSumOfErrors(
            reward_weights, normed, observed_states=None, gamma=0.9, reward_power=power
        )
        # setup_physical_system(motor_type, converter_type)
        physical_system = tu.DummyPhysicalSystem(2)
        # setup_reference_generator('SinusReference', physical_system)
        reference_generator = tu.DummyReferenceGenerator()
        monkeypatch.setattr(ws, "set_state_array", mock_set_state_array)
        reference_generator._referenced_states = mock_ref_states  # parametrized to [true/false]
        swsoe_obj.set_modules(physical_system, reference_generator)

        # Test if the mock_set_state_array function is called only 2 times
        assert self._mock_set_state_array_counter == expected_call_counter  # 2
        # Test the reward weights and range
        if normed:
            assert swsoe_obj.reward_range == expected_reward_range_normed  # (0, 1)
            for val in swsoe_obj._reward_weights:
                assert val == expected_reward   # 1 / 3

        else:
            assert swsoe_obj.reward_range == expected_reward_range  # (0, 3)
            for val in swsoe_obj._reward_weights:
                assert val == expected_reward_normed  # 1
