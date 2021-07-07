from gym_electric_motor.physical_systems.noise_generators import *
import gym_electric_motor.physical_systems.noise_generators as ng
import numpy as np
import pytest


class TestNoiseGenerator:
    """Class to test the basic functions of each NoiseGenerator"""

    # defined test values
    _state_variables = ['omega', 'torque', 'u', 'i']
    _test_state_name = ['omega', 'torque', 'u_a', 'u_e', 'i_a', 'i_e']
    _test_signal_powers = [0.7, 0.6, 0.5, 0.4]
    _state_length = 4  # Number of elements in _state_variables
    _test_noise = np.ones(_state_length) * 0.25
    _test_object = NoiseGenerator()

    # counter
    _monkey_noise_counter = 0
    _monkey_set_state_names = 0
    _monkey_set_state_array_none = 0

    def monkey_noise(self, *_, **__):
        """
        mock function for noise()
        :param _:
        :param __:
        :return:
        """
        self._monkey_noise_counter += 1
        return self._test_noise

    def monkey_set_state_array(self, values, names):
        """
        mock function for utils.set_state_array()
        :param values:
        :param names:
        :return:
        """
        assert len(values) == len(names), 'different number of values and names'
        assert values == self._test_signal_powers, 'values are not the test signal powers'
        assert names == self._state_variables, 'names are not the state variables'
        return np.array(values)

    def monkey_set_state_array_none(self, values, names):
        """
        mock function for utils.set_state_array()
        This function is for the case if names should be none
        :param values:
        :param names:
        :return:
        """
        self._monkey_set_state_array_none += 1
        assert names is None
        return np.array(values)

    def monkey_set_state_names(self, state_names):
        """
        mock function for set_state_names
        :param state_names:
        :return:
        """
        self._monkey_set_state_names += 1
        assert state_names == self._state_variables, 'names are not the state variables'

    def test_noise(self, monkeypatch):
        """
        test noise()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        self._test_object = NoiseGenerator()
        monkeypatch.setattr(NoiseGenerator, "_state_variables", self._state_variables)
        # call function to test
        test_noise = self._test_object.noise()
        # verify the expected results
        assert all(test_noise == np.zeros(self._state_length)), 'noise is not zero'

    def test_reset(self, monkeypatch):
        """
        test reset()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        self._test_object = NoiseGenerator()
        self._test_noise = np.ones(self._state_length) * 0.25
        monkeypatch.setattr(NoiseGenerator, "noise", self.monkey_noise)
        # call function to test
        test_value = self._test_object.reset()
        # verify the expected results
        assert all(test_value == self._test_noise), 'noise is not as expected'
        assert self._monkey_noise_counter == 1, '_noise() not called once'

    def test_set_state_names(self):
        """
        test set_state_names()
        :return:
        """
        # call function to test

        self._test_object.set_state_names(self._test_state_name)
        # verify the expected results
        assert self._test_object._state_variables == self._test_state_name, 'names are not the state variables'

    def test_set_signal_power(self, monkeypatch):
        """
        test set_signal_power()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        self._test_object = NoiseGenerator()
        monkeypatch.setattr(NoiseGenerator, "_state_variables", self._state_variables)
        monkeypatch.setattr(ng, "set_state_array", self.monkey_set_state_array)
        # call function to test
        self._test_object.set_signal_power_level(self._test_signal_powers)
        # verify the expected results
        assert all(self._test_object._signal_powers == np.array(self._test_signal_powers)), \
            'signal powers are not set correctly'

        # test for warning
        # setup test scenario
        monkeypatch.setattr(NoiseGenerator, "_state_variables", None)
        monkeypatch.setattr(ng, "set_state_array", self.monkey_set_state_array_none)
        with pytest.warns(UserWarning):
            # call function to test
            self._test_object.set_signal_power_level(self._test_signal_powers)
            # verify the expected results
        assert all(self._test_object._signal_powers == np.array(self._test_signal_powers)), \
            'signal powers are not set correctly'


class TestGaussianWhiteNoiseGenerator(TestNoiseGenerator):
    """Class to test the GaussianWhiteNoiseGenerator"""

    # defined test values
    _default_noise_levels = 0.0
    _default_noise_length = 10000
    _noise_length = 10
    _noise_levels = None
    _state_length = 4
    _test_noise = np.ones((_noise_length, _state_length)) * 0.42

    # counter
    _monkey_set_state_array_noise_levels_counter = 0
    _monkey_noise_generator_counter = 0

    def monkey_set_state_array_noise_levels(self, values, names):
        """
        mock function for set_state_array()
        :param values:
        :param names:
        :return:
        """
        self._monkey_set_state_array_noise_levels_counter += 1
        assert values == self._noise_levels, 'values are not the noise levels'
        assert names == self._state_variables, 'names are not the state variables'
        if type(values) is list:
            assert len(values) == len(names), 'different number of values and names'
            result = np.array(values)
        elif type(values) is dict:
            assert len(values) == len(names), 'different number of values and names'
            result = np.array(list(values.values()))
        elif type(values) is np.ndarray:
            assert len(values) == len(names), 'different number of values and names'
            result = values
        elif type(values) is float:
            result = values * np.ones(len(names))
        else:
            raise Exception("Not existing type")
        return result

    def monkey_noise_generator(self):
        """
        mock function for noise generator()
        :return:
        """
        self._monkey_noise_generator_counter += 1
        self._test_object._noise = np.ones((self._noise_length, self._state_length))
        self._test_object._noise_pointer = 0

    def test_init_default(self):
        """
        test initialization without parameters
        :return:
        """
        # call function to test
        self._test_object = GaussianWhiteNoiseGenerator()
        # verify the expected results
        assert self._test_object._noise_levels == self._default_noise_levels,\
            'initialized noise levels are not the default ones'
        assert self._test_object._noise_length == self._test_object._noise_pointer == self._default_noise_length, \
            'initialized noise length is not the default one'

    @pytest.mark.parametrize("noise_levels", [0.1, [0.1, 0.3, 0.2, 0.5],
                                              dict(omega=0.1, torque=0.4, u=0.2, i=0.6)])
    @pytest.mark.parametrize("noise_length", [_noise_length])
    def test_init_parametrized(self, noise_levels, noise_length):
        """
        test parametrized initialization
        :param noise_levels: possible noise levels
        :param noise_length: possible noise length
        :return:
        """
        # call function to test
        self._test_object = GaussianWhiteNoiseGenerator(noise_levels, noise_length)
        # verify the expected results
        assert noise_levels == self._test_object._noise_levels, 'noise levels are not passed correctly'
        assert self._test_object._noise_length == self._test_object._noise_pointer == noise_length,\
            'noise length is not passed correctly'

    @pytest.mark.parametrize("noise_levels", [0.1, [0.1, 0.3, 0.2, 0.5],
                                              dict(omega=0.1, torque=0.4, u=0.2, i=0.6)])
    def test_set_state_names(self, monkeypatch, noise_levels):
        """
        test set state names
        :param monkeypatch:
        :param noise_levels: possible noise levels
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(NoiseGenerator, "set_state_names", self.monkey_set_state_names)
        self._test_object = GaussianWhiteNoiseGenerator()
        monkeypatch.setattr(self._test_object, "_noise_levels", noise_levels)
        monkeypatch.setattr(ng, "set_state_array", self.monkey_set_state_array_noise_levels)
        self._noise_levels = noise_levels
        # call function to test
        self._test_object.set_state_names(self._state_variables)
        # verify the expected results
        assert type(self._test_object._noise_levels) is np.ndarray
        assert self._monkey_set_state_array_noise_levels_counter == 1, 'set_state_array() not called once'
        assert self._monkey_set_state_names == 1, 'set_state_names not called once'

    @pytest.mark.parametrize("noise_pointer, result_noise_pointer, count_generate_noise",
                             [(0, 1, 0), (1, 2, 0), (3, 4, 0), (8, 9, 0), (9, 10, 0), (10, 1, 1)])
    def test_noise(self, monkeypatch, noise_pointer, result_noise_pointer, count_generate_noise):
        """
        test noise()
        :param monkeypatch:
        :param noise_pointer: current step in the noise array
        :param result_noise_pointer: next step in the noise array
        :param count_generate_noise: counter for generate_noise()
        :return:
        """
        # setup test scenario
        self._test_object = GaussianWhiteNoiseGenerator()
        monkeypatch.setattr(self._test_object, "_noise_length", self._noise_length)
        monkeypatch.setattr(self._test_object, "_noise_pointer", noise_pointer)
        monkeypatch.setattr(self._test_object, "_noise", self._test_noise)
        monkeypatch.setattr(GaussianWhiteNoiseGenerator, "_generate_noise", self.monkey_noise_generator)
        # call function to test
        noise = self._test_object.noise()
        # verify the expected results
        assert len(noise) == self._state_length, 'Wrong noise length'
        assert self._test_object._noise_pointer == result_noise_pointer, ' unexpected noise pointer'
        assert self._monkey_noise_generator_counter == count_generate_noise, 'noise_generator() called unexpected often'
