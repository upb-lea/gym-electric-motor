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
