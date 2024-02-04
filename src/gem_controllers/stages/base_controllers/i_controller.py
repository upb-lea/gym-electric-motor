import numpy as np

from ..base_controllers import BaseController


class IController(BaseController):
    """This class represents an integration controller, which can be combined e.g. with a proportional controller to a
    PI controller.
    """

    # (float): Additional term to avoid division by zero
    epsilon = 1e-6

    @property
    def i_gain(self):
        """I gain of the I controller"""
        return self._i_gain

    @i_gain.setter
    def i_gain(self, value):
        # i_gain is at least zero, to avoid unstable behavior
        value = np.clip(value, 0.0, np.inf)
        self._i_gain = value

    @property
    def tau(self):
        """Sampling time of the system"""
        return self._tau

    @tau.setter
    def tau(self, value: [float, int]):
        self._tau = float(value)

    @property
    def action_range(self):
        """Action range of the base controller"""
        return self._action_range

    @action_range.setter
    def action_range(self, value):
        self._action_range = value

    @property
    def state_indices(self):
        """Indices of the controlled states"""
        return self._state_indices

    @state_indices.setter
    def state_indices(self, value):
        self._state_indices = np.array(value)

    @property
    def integrator(self):
        """Integrated value of the I controller"""
        return self._integrator

    @integrator.setter
    def integrator(self, value):
        self._integrator = value

    def __init__(self, control_task):
        """
        Args:
            control_task(str): Control task of the base controller
        """

        super().__init__(control_task)
        self._state_indices = np.array([])
        self._action_range = (np.array([]), np.array([]))
        self.i_gain = np.array([])
        self._integrator = np.array([0.0])
        self._tau = None
        self._clipped = np.array([])

    def __call__(self, state, reference):
        """
        Calculate the reference values of the I controller

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
             np.array: reference values of the next stage
        """
        return self.control(state, reference)

    def _control(self, _state, _reference):
        """Calculate the reference for the underlying stage"""
        return self._i_gain * self._integrator

    def control(self, state, reference):
        """
        Calculate the reference for the underlying stage

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
             np.array: reference values of the next stage
        """
        return self._control(state[self._state_indices], reference)

    def integrate(self, state, reference):
        """
        Integrates the control error.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.
        """

        error = reference - state
        self._integrator = self._integrator + (error * self._tau * ~self._clipped)

    def reset(self):
        """Reset the integrated values"""
        super().reset()
        self._integrator = np.zeros_like(self._i_gain)
        self._clipped = np.zeros_like(self._i_gain, dtype=bool)
