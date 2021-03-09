import numpy as np

from .mechanical_load import MechanicalLoad


class OrnsteinUhlenbeckLoad(MechanicalLoad):
    """ External speed mechanical load system which will set the speed to a predefined speed-function/ speed-profile.
    """

    HAS_JACOBIAN = False

    def __init__(self, mu=0, sigma=1e-4, theta=1, tau=1e-4, omega_range=(-200.0, 200.0), **kwargs):
        """
        Args:
            sigma_range(2-tuple(float)): Lower and upper bound that the
            tau(float): discrete time step of the system
            kwargs(float): further arguments passed to the superclass :py:class:`.MechanicalLoad`
        """
        super().__init__(**kwargs)
        self._omega = np.array([0])
        self.theta = theta
        self.mu = mu
        self.tau = tau
        self.sigma = sigma
        self._omega_range = omega_range

    def mechanical_ode(self, t, mechanical_state, torque):
        omega = mechanical_state
        max_diff = (self._omega_range[1] - omega) / self.tau
        min_diff = (self._omega_range[0] - omega) / self.tau
        diff = self.theta * (self.mu - omega) * self.tau \
            + self.sigma * np.sqrt(self.tau) * np.random.normal(size=1)
        np.clip(diff, min_diff, max_diff, out=diff)
        return diff

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._omega = np.array([0.0])
        return self._omega
