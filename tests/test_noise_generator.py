from gym_electric_motor.physical_systems.noise_generators import *
import numpy as np
import pytest

# test parameter
g_noise_levels = np.array([1, 10, 0.5, 0, 1])
g_state_variables = ['omega', 'torque', 'i', 'u', 'u_sup']
g_signal_powers = np.array([1, 2, 3, 4, 5])


def test_gaussian_white_noise_generator():
    """
    This function test the Gaussian White Noise Generator:
    Following functions are included:
    NoiseGenerator
        reset
        noise
        set_state_variables
        set_signal_power_level

    GaussianWhiteNoiseGenerator
        _noise
        noise
        _generate_noise
    :return:
    """
    # set random number generator to fixed initial value for testing
    np.random.seed(123)
    # pre calculated and expected results
    test_results = np.array([[1.65143654, -48.53358487, -0.64336894, 0., -4.33370201],
                            [-0.67888615, -1.89417938, 2.23708444, 0., -2.2199098],
                            [-0.43435128, 44.11860165, 3.28017913, 0., 1.930932]])
    noise_length = 15
    # test default initialization
    noise_generator_default = GaussianWhiteNoiseGenerator()
    # test full initialization
    noise_generator_init = GaussianWhiteNoiseGenerator(g_noise_levels, noise_length)
    # test set state variables function
    noise_generator_init.set_state_names(g_state_variables)
    assert (noise_generator_init._state_variables == g_state_variables), " Wrong state variables setting"
    # test set signal power level function
    noise_generator_init.set_signal_power_level(g_signal_powers)
    # test reset function
    noise_generator_init.reset()
    # test noise generator
    for k in range(3):
        noise = noise_generator_init.noise()
        assert sum((noise-test_results[k])**2) < 1e-10, " Noise test gone wrong"
    # verify that the output is zero if all signal power levels are set to zero
    zeros_expected = np.zeros(5)
    noise_generator_init.set_signal_power_level(zeros_expected)
    # assert all(noise_generator_init.reset() == zeros_expected)
    # assert all(noise_generator_init.noise() == zeros_expected)

    # initialization with one value
    noise_generator_init = GaussianWhiteNoiseGenerator(0.0)
    # test if an TypeError is raised if signal powers should be set before state variables
    with pytest.raises(TypeError):
        noise_generator_init.set_signal_power_level(g_signal_powers)

    noise_generator_init.set_state_names(g_state_variables)
    noise_generator_init.set_signal_power_level(g_signal_powers)
    assert all(noise_generator_init.reset() == zeros_expected)
