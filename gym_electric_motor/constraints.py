import numpy as np

from .utils import set_state_array


class Constraint:

    def __call__(self, state):
        raise NotImplementedError

    def set_modules(self, ps):
        pass


class LimitConstraint(Constraint):

    def __init__(self, observed_state_names='all'):
        self._observed_state_names = observed_state_names
        self._limits = None
        self._observed_states = None

    def __call__(self, state):
        return float(any(state[self._observed_states] > 1.0))

    def set_modules(self, ps):
        self._limits = ps.limits
        if self._observed_state_names == 'all':
            self._observed_state_names = ps.state_names
        if self._observed_state_names is None:
            self._observed_state_names = []
        self._observed_states = set_state_array(
            dict.fromkeys(self._observed_state_names, 1), ps.state_names
        ).astype(bool)


class SquaredConstraint(Constraint):

    def __init__(self, states=()):
        self._states = states
        self._state_indices = ()
        self._limits = ()
        self._normalized = False

    def set_modules(self, ps):
        self._state_indices = [ps.state_positions[state] for state in self._states]
        self._limits = ps.limits[self._state_indices]
        self._normalized = not np.all(ps.state_space.high[self._state_indices] == self._limits)

    def __call__(self, state):
        state_ = state[self._state_indices] if self._normalized else state[self._state_indices] / self._limits
        return float(np.sum(state_**2) > 1.0)
