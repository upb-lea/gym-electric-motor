import numpy as np
import warnings

from ..random_component import RandomComponent
from ..utils import set_state_array


class NoiseGenerator(RandomComponent):
    """
    The noise generator generates noise added to the state for the observation.
    """

    _state_variables = None
    _signal_powers = None

    def reset(self):
        """
        Reset of all internal states of the Noise generator to a default.

        Returns:
             Noise for the initial observation at time 0.
        """
        self.next_generator()
        return self.noise()

    def noise(self, *_, **__):
        """
        Call this function to get the additive noise.

        Returns:
             ndarray(float): Array of noise added to each state.
        """
        return np.zeros_like(self._state_variables, dtype=float)

    def set_state_names(self, state_names):
        """
        Announcement of the state names by the physical system to the noise generator.

        Args:
            state_names(list(str)): List of state names of the physical system.
        """
        self._state_variables = state_names

    def set_signal_power_level(self, signal_powers):
        """
        Setting of the signal power levels (Nominal values) as baseline for the noise power.

        Args:
            signal_powers(list(float)): Array of signal powers (nominal values).
        """
        if self._state_variables is None:
            warnings.warn(UserWarning("No state variable set"))
        self._signal_powers = set_state_array(signal_powers, self._state_variables)


class GaussianWhiteNoiseGenerator(NoiseGenerator):
    """
    Noise Generator that adds gaussian distributed additional noise with zero mean and selectable power.
    """
    _noise = None

    def __init__(self, noise_levels=0.0, noise_length=10000):
        """
        Args:
            noise_levels(dict/list/ndarray(float)): Fraction of noise power over the signal powers.
            noise_length(float): Length of simultaneously generated noise points for speed up.
        """
        RandomComponent.__init__(self)
        self._noise_levels = noise_levels
        self._noise_length = noise_length
        self._noise_pointer = noise_length

    def set_state_names(self, state_names):
        # Docstring of superclass
        super().set_state_names(state_names)
        self._noise_levels = set_state_array(self._noise_levels, state_names)

    def noise(self, *_, **__):
        # Docstring of superclass
        if self._noise_pointer == self._noise_length:
            self._generate_noise()
        noise = self._noise[self._noise_pointer]
        self._noise_pointer += 1
        return noise

    def _generate_noise(self):
        """
        Helper function to generate noise in batches of self._noise_length steps to avoid calculating every steps and
        to speed up the computation.
        """
        self._noise = self._random_generator.normal(
            0, self._noise_levels * self._signal_powers, (self._noise_length, len(self._signal_powers))
        )
        self._noise_pointer = 0
