from .pi_controller import PIController
from .continuous_controller import DController


class PIDController(PIController, DController):
    """The PID-Controller is a combination of the PI-Controller and the base P-Controller."""

    def __init__(self, environment, p_gain=5, i_gain=5, d_gain=0.005, param_dict={}, **controller_kwargs):
        p_gain = param_dict.get('p_gain', p_gain)
        i_gain = param_dict.get('i_gain', i_gain)
        d_gain = param_dict.get('d_gain', d_gain)

        PIController.__init__(self, environment, p_gain, i_gain)
        DController.__init__(self, d_gain)

    def control(self, state, reference):
        action = PIController.control(self, state, reference) + self.d_gain * (
                reference - state - self.e_old) / self.tau
        self.e_old = reference - state
        return action

    def reset(self):
        PIController.reset(self)
        self.e_old = 0