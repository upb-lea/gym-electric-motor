import numpy as np

from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class StepReferenceGenerator(SubepisodedReferenceGenerator):
    """
    Reference Generator that generates a step function with a random amplitude, frequency, phase and offset.
    The reference is generated for a certain length and then new parameters are drawn uniformly from a selectable range.
    """
    _amplitude = 0
    _frequency = 0
    _offset = 0

    def __init__(self, amplitude_range=None, frequency_range=(1, 10), offset_range=None, **kwargs):
        """
        Args:
            amplitude_range(tuple(float,float)): Lower and upper limit for the amplitude.
            frequency_range(tuple(float,float)): Lower and upper limit for the frequency.
            offset_range(tuple(float,float)): Lower and upper limit for the offset
            kwargs(any): Arguments passed to the superclass SubepisodedReferenceGenerator .
        """
        super().__init__(**kwargs)
        self._amplitude_range = amplitude_range or (0, np.inf)
        self._frequency_range = frequency_range
        self._offset_range = offset_range or (-np.inf, np.inf)

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        self._amplitude_range = np.clip(self._amplitude_range, 0, (self._limit_margin[1] - self._limit_margin[0]) / 2)
        self._offset_range = np.clip(self._offset_range, self._limit_margin[0], self._limit_margin[1])

    def _reset_reference(self):
        self._amplitude = self._get_current_value(self._amplitude_range)
        self._frequency = self._get_current_value(self._frequency_range).reshape(-1, 1)
        self._offset = np.clip(np.vstack(self._physical_system.n_prll_envs * [self._offset_range.reshape(1, -1)]),
                               a_min=-self._limit_margin[1] + self._amplitude,
                               a_max=self._limit_margin[1] - self._amplitude)
        self._offset = self._random_generator.uniform(self._offset[:, 0], self._offset[:, 1])
        high_low_ratio = self.random_generator.triangular(0, 0.5, 1, size=self._physical_system.n_prll_envs)
        t = np.linspace(0, (self._current_episode_length - 1) * self._physical_system.tau, self._current_episode_length)
        x = self._frequency * (t.reshape(1, -1) % (1/self._frequency))
        x -= high_low_ratio.reshape(-1, 1)
        x = np.sign(x)
        phase = self.random_generator.uniform(size=self._physical_system.n_prll_envs).reshape(-1, 1)
        steps_per_period = 1 / self._frequency / self._physical_system.tau
        # source:
        #  https://stackoverflow.com/questions/20360675/roll-rows-of-a-matrix-independently
        row_idx, col_idx = np.ogrid[:x.shape[0], :x.shape[1]]
        row_rolls = (steps_per_period * phase).astype(int)
        row_rolls[row_rolls <0] += x.shape[1]  # always use negative shift
        col_idx -= row_rolls.reshape(-1, 1)
        x = x[row_idx, col_idx]
        #x = np.roll(x, int(steps_per_period * phase), axis=1)
        self._reference = self._amplitude.reshape(-1, 1) * x + self._offset.reshape(-1, 1)
        self._reference = np.clip(self._reference, self._limit_margin[0], self._limit_margin[1])
