import numpy as np
from scipy import signal as sg
from .subepisoded_reference_generator import SubepisodedReferenceGenerator


class TriangularReferenceGenerator(SubepisodedReferenceGenerator):
    """
    Reference Generator that generates a triangular waveform with random asymmetry. The Amplitude, frequency and offset
    are also randomized within a specified range.
    The reference is generated for a certain length and then new parameters are drawn uniformly from a selectable range.
    """

    def __init__(self, amplitude_range=None, frequency_range=(1, 10), offset_range=None, *_, **kwargs):
        """
        Args:
            amplitude_range(tuple(float,float)): Lower and upper limit for the amplitude.
            frequency_range(tuple(float,float)): Lower and upper limit for the frequency.
            offset_range(tuple(float,float)): Lower and upper limit for the offset
            kwargs(dict): Arguments passed to the superclass SubepisodedReferenceGenerator .
        """
        super().__init__(**kwargs)
        self._amplitude = 0.0
        self._frequency = 0.0
        self._offset = 0.0
        self._amplitude_range = amplitude_range or (0, np.inf)
        self._frequency_range = frequency_range
        self._offset_range = offset_range or (-np.inf, np.inf)

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        # but amplitude and offset cannot exceed limit margin
        self._amplitude_range = np.clip(
            self._amplitude_range, 0, (self._limit_margin[1] - self._limit_margin[0]) / 2
        )
        self._offset_range = np.clip(self._offset_range, self._limit_margin[0], self._limit_margin[1])

    def _reset_reference(self):
        # get absolute values of amplitude, frequency and offset
        self._amplitude = self._get_current_value(self._amplitude_range)
        self._frequency = self._get_current_value(self._frequency_range)
        offset_range = np.clip(
            self._offset_range, -self._limit_margin[1] + self._amplitude,  self._limit_margin[1] - self._amplitude
        )
        self._offset = self._get_current_value(offset_range)

        t = np.linspace(0, (self._current_episode_length - 1) * self._physical_system.tau, self._current_episode_length)
        phase = self._random_generator.uniform() * 2 * np.pi  # note: in the scipy implementation of sawtooth() 1 time-period
        # corresponds to a phase of 2pi
        ref_width = self._random_generator.uniform()  # a random value between 0,1 that creates asymmetry in the triangular reference
        # wave ref_width=1 creates a sawtooth waveform
        self._reference = self._amplitude * sg.sawtooth(2*np.pi * self._frequency * t + phase, ref_width) + self._offset
        self._reference = np.clip(self._reference, self._limit_margin[0], self._limit_margin[1])
