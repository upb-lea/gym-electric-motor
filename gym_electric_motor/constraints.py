import numpy as np

from .utils import set_state_array


class Constraint:
    """Base class for all constraints in the ConstraintMonitor."""

    def __call__(self, state):
        """Function that is called to check the constraint.

        Args:
            state(numpy.ndarray(float)): The current physical systems state.

        Returns:
              float in [0.0, 1.0]: Degree how much the constraint has been violated.
                0.0: No violation
                (0.0, 1.0): Undesired zone near to a full violation. No episode termination.
                1.0: Full violation and episode termination.
        """
        raise NotImplementedError

    def set_modules(self, ps):
        """Called by the environment that the Constraint can read information from the PhysicalSystem.

        Args:
            ps(PhysicalSystem): PhysicalSystem of the environment.
        """
        pass


class LimitConstraint(Constraint):
    """Constraint to observe the limits on one or more system states.

    This constraint observes if any of the systems state values exceeds the limit specified in the PhysicalSystem.

    .. math::
        1.0 >= s_i / s_{i,max}

    For all :math:`i` in the set of PhysicalSystems states :math:`S`.

    """

    def __init__(self, observed_state_names='all_states'):
        """
        Args:
            observed_state_names(['all_states']/iterable(str)): The states to observe. \n
                - ['all_states']: Shortcut for observing all states.
                - iterable(str): Pass an iterable containing all state names of the states to observe.
        """
        self._observed_state_names = observed_state_names
        self._limits = None
        self._observed_states = None

    def __call__(self, state):
        observed = state[self._observed_states]
        violation = any(abs(observed) > 1.0)
        return float(violation)

    def set_modules(self, ps):
        self._limits = ps.limits
        if 'all_states' in self._observed_state_names:
            self._observed_state_names = ps.state_names
        if self._observed_state_names is None:
            self._observed_state_names = []
        self._observed_states = set_state_array(
            dict.fromkeys(self._observed_state_names, 1), ps.state_names
        ).astype(bool)


class SquaredConstraint(Constraint):
    """A squared constraint on multiple states as it is required oftentimes for the dq-currents in synchronous and
    asynchronous electric motors.

    .. math::
        1.0 <= \sum_{i \in S} (s_i / s_{i,max})^2

    :math:`S`: Set of the observed PhysicalSystems states
    """

    def __init__(self, states=()):
        """
        Args:
            states(iterable(str)): Names of all states to be observed within the SquaredConstraint.
        """
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
