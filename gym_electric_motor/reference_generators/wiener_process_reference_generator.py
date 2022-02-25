import numpy as np

from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class WienerProcessReferenceGenerator(SubepisodedReferenceGenerator):
    """Reference Generator that generates a reference for one state by a Wiener Process with the changing parameter
    sigma and mean = 0.
    """

    def __init__(self, sigma_range=(1e-3, 1e-1), initial_range=None, **kwargs):
        """
        Args:
            sigma_range(Tuple(float,float)): Lower and Upper limit for the sigma-parameter of the WienerProcess.
            initial_range(Tuple(float,float)): Minimal and maximal normalized value of the initial reference point
             of a new episode. Default: Equal to the limit margin inherited from the
             :py:class:`.SubepisodedReferenceGenerator`
            kwargs: Further arguments to pass to :py:class:`.SubepisodedReferenceGenerator`
        """
        super().__init__(**kwargs)
        self._initial_range = initial_range
        self._current_sigma = 0
        self._sigma_range = sigma_range

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        if self._initial_range is None:
            self._initial_range = self._limit_margin

    def _reset_reference(self):
        self._current_sigma = 10 ** self._get_current_value(np.log10(self._sigma_range))
        # (#batchsize, #episode_length)
        random_values = self._random_generator.normal(0, self._current_sigma,
                                                      size=(self._current_episode_length, self._current_sigma.size)).T
        self._reference = np.zeros_like(random_values)
        self._reference = np.clip(random_values + self._reference_value.reshape(-1, 1),
                                  a_min=self._limit_margin[0], a_max=self._limit_margin[1])

    def reset(self, initial_state=None, initial_reference=None):
        if initial_reference is None:
            initial_reference = np.zeros((self._physical_system.n_prll_envs, len(self._referenced_states)), dtype=float)
            initial_reference[:, self._referenced_states] =\
                self.random_generator.uniform(self._initial_range[0], self._initial_range[1], size=self._physical_system.n_prll_envs)
        return super().reset(initial_state, initial_reference)
