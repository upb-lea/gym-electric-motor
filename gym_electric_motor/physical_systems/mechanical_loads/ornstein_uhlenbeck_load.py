import numpy as np

from .mechanical_load import MechanicalLoad


class OrnsteinUhlenbeckLoad(MechanicalLoad):
    """ External speed mechanical load system which will set the speed to a predefined speed-function/ speed-profile.
    """

    HAS_JACOBIAN = False

    def __init__(self, mu=0, sigma=1e-4, theta=1, tau=1e-4, **kwargs):
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

    def mechanical_ode(self, t, mechanical_state, torque):
        diff = self.theta * (self.mu - self._omega) * self.tau \
               + self.sigma * np.sqrt(self.tau) * np.random.normal(size=1)
        self._omega = self._omega + diff * self.tau
        return diff

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._omega = np.array([0.0])
        return self._omega
