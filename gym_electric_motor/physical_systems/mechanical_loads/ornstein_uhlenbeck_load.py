import numpy as np

from .mechanical_load import MechanicalLoad


class OrnsteinUhlenbeckLoad(MechanicalLoad):
    """The Ornstein-Uhlenbeck Load sets the speed to a torque-independent signal specified by the underlying OU-Process.
    """

    HAS_JACOBIAN = False

    def __init__(self, mu=0, sigma=1e-4, theta=1, tau=1e-4, omega_range=(-200.0, 200.0), **kwargs):
        """
        Args:
            mu(float): Mean value of the underlying gaussian distribution of the OU-Process.
            sigma(float): Standard deviation of the underlying gaussian distribution of the  OU-Process.
            theta(float): Drift towards the mean of the OU-Process.
            tau(float): discrete time step of the system
            omega_range(2-Tuple(float)): Minimal and maximal value for the process.
            kwargs(dict): further arguments passed to the superclass :py:class:`.MechanicalLoad`
        """
        super().__init__(**kwargs)
        self._omega = np.random.uniform(self._omega_range[0], self._omega_range[1], 1)
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
        self._omega = np.random.uniform(self._omega_range[0], self._omega_range[1], 1)
        return self._omega
