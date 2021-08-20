import numpy as np

from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class WienerProcessReferenceGenerator(SubepisodedReferenceGenerator):
    """Reference Generator that generates a reference for one state by a Wiener Process with the changing parameter
    sigma and mean = 0.
    """

    def __init__(self, sigma_range=(1e-3, 1e-1), **kwargs):
        """
        Args:
            sigma_range(Tuple(float,float)): Lower and Upper limit for the sigma-parameter of the WienerProcess.
            kwargs: Further arguments to pass to SubepisodedReferenceGenerator
        """
        super().__init__(**kwargs)

        self._current_sigma = 0
        self._sigma_range = sigma_range

    def _reset_reference(self):
        self._current_sigma = 10 ** self._get_current_value(np.log10(self._sigma_range))
        random_values = self._random_generator.normal(0, self._current_sigma, self._current_episode_length)
        self._reference = np.zeros_like(random_values)
        reference_value = self._reference_value
        for i in range(self._current_episode_length):
            reference_value += random_values[i]
            if reference_value > self._limit_margin[1]:
                reference_value = self._limit_margin[1]
            if reference_value < self._limit_margin[0]:
                reference_value = self._limit_margin[0]
            self._reference[i] = reference_value
