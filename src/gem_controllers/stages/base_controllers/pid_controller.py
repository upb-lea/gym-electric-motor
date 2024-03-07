import numpy as np

from .pi_controller import PIController


class PIDController(PIController):
    """This class extends the PI controller by differential controller."""

    @property
    def d_gain(self):
        """D gain of the D base controller"""
        return self._d_gain

    @d_gain.setter
    def d_gain(self, value):
        self._d_gain = np.array(value)
        self._last_error = np.zeros_like(self._d_gain, dtype=float)

    def __init__(self, control_task):
        """
        Args:
            control_task(str): Control task of the PID controller
        """
        super().__init__(control_task)
        self._d_gain = np.array([])
        self._last_error = np.array([])

    def control(self, state, reference):
        """
        Calculate the action of the PID controller.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            action(np.ndarray): The action of the PID controller
        """
        pi_action = super().control(state, reference)
        current_error = reference - state[self.state_indices]
        d_action = self._d_gain * (current_error - self._last_error) / self._tau
        self._last_error = current_error
        return pi_action + d_action

    def _tune_current_controller(self, env, env_id, a=4):
        """Set the gains of the P-, I- and D-component for the current control.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetrical optimum.
        """

        super()._tune_current_controller(env, env_id, a)
        self.d_gain = self.p_gain * self.tau

    def _tune_speed_controller(self, env, env_id, a=4, t_n=None):
        """Set the gains of the P-, I- and D-component for the speed control.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetrical optimum.
            t_n(float): Time constant of the underlying torque controller.
        """

        super()._tune_speed_controller(env, env_id, a)
        self.d_gain = self.p_gain * self.tau
