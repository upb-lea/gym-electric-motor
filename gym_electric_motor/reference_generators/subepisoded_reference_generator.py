import numpy as np
from gym.spaces import Box

from ..random_component import RandomComponent
from ..core import ReferenceGenerator
from ..utils import set_state_array


class SubepisodedReferenceGenerator(ReferenceGenerator, RandomComponent):
    """Base Class for Reference Generators, which change their parameters in certain ranges after a random number of
    time steps and can pre-calculate their references in these "sub episodes".
    """

    def __init__(self, reference_state='omega', episode_lengths=(500, 2000), limit_margin=None, **kwargs):
        """
        Args:
            reference_state(str): Name of the state that this reference generator is referencing.
            episode_lengths(Tuple(int,int)): Minimum and maximum length of a sub episode.
            limit_margin(Tuple(float,float)/float/None):
                Factor, how close the references should get to the limits.
                If a tuple is passed, then the lower[0] and upper[1] margin might differ.
                If a float is passed, both margins are equal.
                If None(default), the limit margin equals (nominal values/limits).
                In general, the limit margin should not exceed (-1, 1)
            kwargs(dict): Keyword arguments to be passed to the base class ReferenceGenerator
        """
        ReferenceGenerator.__init__(self, **kwargs)
        RandomComponent.__init__(self)
        self.reference_space = Box(-1, 1, shape=(1,), dtype=np.float64)
        self._reference = None
        self._limit_margin = limit_margin
        self._reference_value = 0.0
        self._reference_state = reference_state.lower()
        self._episode_len_range = episode_lengths
        self._current_episode_length = int(self._get_current_value(episode_lengths))
        self._k = 0
        self._reference_names = [self._reference_state]

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        self._referenced_states = set_state_array(
            {self._reference_state: 1}, physical_system.state_names
        ).astype(bool)
        rs = self._referenced_states
        ps = physical_system
        if self._limit_margin is None:
            upper_margin = (ps.nominal_state[rs] / ps.limits[rs])[0] * ps.state_space.high[rs]
            lower_margin = (ps.nominal_state[rs] / ps.limits[rs])[0] * ps.state_space.low[rs]
            self._limit_margin = lower_margin[0], upper_margin[0]
        elif type(self._limit_margin) in [float, int]:
            upper_margin = self._limit_margin * ps.state_space.high[rs]
            lower_margin = self._limit_margin * ps.state_space.low[rs]
            self._limit_margin = lower_margin[0], upper_margin[0]
        elif type(self._limit_margin) is tuple:
            lower_margin = self._limit_margin[0] * ps.state_space.low[rs]
            upper_margin = self._limit_margin[1] * ps.state_space.high[rs]
            self._limit_margin = lower_margin[0], upper_margin[0]
        else:
            raise Exception('Unknown type for the limit margin.')
        self.reference_space = Box(lower_margin[0], upper_margin[0], shape=(1,), dtype=np.float64)

    def reset(self, initial_state=None, initial_reference=None):
        """
        The references are reset. If an initial reference is passed, this value will be the first reference value of
        the next episode. Otherwise it will be 0.

        Args:
            initial_state(ndarray(float)): The initial state of the physical system.
            initial_reference(ndarray(float)): (Optional) The first reference value.

        Returns:
             initial_reference(ndarray(float)): initial reference array.
             initial_reference_observation(element of reference_space): An initial observation of the next reference.
             trajectory(None): No initial trajectory is passed.
        """
        if initial_reference is not None:
            self._reference_value = initial_reference[self._referenced_states][0]
        else:
            self._reference_value = 0.0
        self.next_generator()
        self._current_episode_length = -1
        return super().reset(initial_state)

    def get_reference(self, *_, **__):
        reference = np.zeros_like(self._referenced_states, dtype=float)
        reference[self._referenced_states] = self._reference_value
        return reference

    def get_reference_observation(self, *_, **__):
        if self._k >= self._current_episode_length:
            self._k = 0
            self._current_episode_length = int(self._get_current_value(self._episode_len_range))
            self._reset_reference()
        self._reference_value = self._reference[self._k]
        self._k += 1
        return np.array([self._reference_value])

    def _reset_reference(self):
        """
        Subclasses implement in this method its generation of the references for the next self._current_episode_length
        time steps and write it into self._reference.
        """
        raise NotImplementedError

    def _get_current_value(self, value_range):
        """
        Return a uniform distributed value for the next sub episode.

        If float or int is passed this value will be returned. Otherwise a uniform distributed value
        between value_range[0] and value_range[1] is returned.
        """
        if type(value_range) in [int, float]:
            return value_range
        elif type(value_range) in [list, tuple, np.ndarray]:
            return (value_range[1] - value_range[0]) * self._random_generator.uniform() + value_range[0]
