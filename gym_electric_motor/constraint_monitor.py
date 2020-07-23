import numpy as np


class ConstraintMonitor:
    """
    The ConstraintMonitor Class monitors the system-states and assesses whether
    they comply the given limits or violate them.
    It returns the necessary information for the RewardFunction, to calculate
    the corresponding reward-value.
    The constraints of the system-states can be generally described by the user
    or are restricted by the physical enviroment-limits.
    The user can predefine a function or a class, that returns a value out of
    [0, 1], where 0 is no violation and 1 is a hard violation.
    The user input have to follow the basic structure, given in the
    userdefined_constraints.py file.
    """

    @property
    def observed_states(self):
        """
        States, which should be monitored

        Returns:
            observed_states(list): monitored states
        """
        return self._observed_states

    def __init__(self,
                 external_monitor=None,
                 normalised=False,
                 *args, **kwargs):
        """
        Args:
            external_monitor(class instance or function): external given
                Class or function, that return by call a bool-value or a float
                out of [0, 1] as feedback for the RewardFunction
            normalised(bool): describes whether the constraints, given by the
                user, are normalised or not
        Return:
        """
        self._external_monitor = external_monitor
        self._observed_states = None
        self._limits = None
        self._normalised = normalised
        self._physical_system = None
        self.args = args
        self.kwargs = kwargs

    def set_modules(self, physical_system, observed_states):
        """
        Setting the necessary attributes for different class-methods

        Args:
            physical_system: instance from physical_system
            observed_states(list): list with boolean entries, indicating which
                states have to be monitored
        Returns:
        """
        self._physical_system = physical_system
        self._limits = physical_system.limits / abs(physical_system.limits)
        self._observed_states = observed_states

    def check_constraint_violation(self, state, k=None):
        """
        Checks if a given system-state violates the constraints

        Args:
            state(np.ndarray): system-state of the environment
            k(int): System momentary time-step

        Returns:
            integer value from [0, 1], where 0 is no violation at all and 1 is
            a hard constraint violation
        """
        if self._external_monitor is not None:
            violation_return = self._external_monitor(
                                        state=state,
                                        observed_states=self._observed_states,
                                        k=k,
                                        physical_system=self._physical_system,
                                        **self.kwargs)
            return float(violation_return)
        else:
            violation_return = (abs(state[self._observed_states]) > self._limits[
                self._observed_states]).any()
            return float(violation_return)

