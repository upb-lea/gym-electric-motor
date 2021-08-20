import numpy as np
from scipy import signal as sg
from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class SawtoothReferenceGenerator(SubepisodedReferenceGenerator):
    """
     Reference Generator that generates a Sawtooth wave with a random amplitude, frequency, phase and offset.
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
            kwargs(dict): Arguments passed to the superclass SubepisodedReferenceGenerator .

        """
        super().__init__(**kwargs)
        self._amplitude_range = amplitude_range or (0, np.inf)
        self._frequency_range = frequency_range
        self._offset_range = offset_range or (-np.inf, np.inf)

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        self._amplitude_range = np.clip(self._amplitude_range, 0, (self._limit_margin[1] - self._limit_margin[0]) / 2)
        # limit_margin will range from(-1, 1) but amplitude cannot exceed 1
        self._offset_range = np.clip(self._offset_range, self._limit_margin[0], self._limit_margin[1])

    def _reset_reference(self):
        # get absolute values of amplitude, frequency and offset
        self._amplitude = self._get_current_value(self._amplitude_range)
        self._frequency = self._get_current_value(self._frequency_range)
        offset_range = np.clip(self._offset_range, -self._limit_margin[1] + self._amplitude,
                               self._limit_margin[1] - self._amplitude)
        self._offset = self._get_current_value(offset_range)

        t = np.linspace(0, (self._current_episode_length - 1) * self._physical_system.tau, self._current_episode_length)
        phase = self.random_generator.uniform() * 2 * np.pi
        # note: in the scipy implementation of sawtooth() 1 time-period corresponds to a phase of 2pi
        self._reference = self._amplitude * sg.sawtooth(2 * np.pi * self._frequency * t + phase) + self._offset
        self._reference = np.clip(self._reference, self._limit_margin[0], self._limit_margin[1])
