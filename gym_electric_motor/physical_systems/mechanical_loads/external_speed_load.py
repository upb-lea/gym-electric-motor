import numpy as np
import warnings

from .mechanical_load import MechanicalLoad


class ExternalSpeedLoad(MechanicalLoad):
    """
       External speed mechanical load system which will set the speed to a
       predefined speed-function/ speed-profile.
    """

    HAS_JACOBIAN = False

    @property
    def omega(self):
        """
        Returns:
            float: Function-value for omega in rad/s at time-step t.
        """
        return self._omega_initial

    def __init__(self, speed_profile, load_initializer=None, tau=1e-4, speed_profile_kwargs=None, **kwargs):
        """
        Args:
            speed_profile(float -> float): A callable(t, **speed_profile_args) -> float
                which takes a timestep t and custom further arguments and returns a speed omega
                example:
                    (lambda t, amplitude, freq: amplitude*numpy.sin(2*pi*f)))
                    with additional parameters:
                        amplitude(float), freq(float), time(float)
            tau(float): discrete time step of the system
            speed_profile_kwargs(dict): further arguments for speed_profile
            kwargs(dict): Arguments to be passed to superclass :py:class:`.MechanicalLoad`

        """
        super().__init__(**kwargs)
        speed_profile_kwargs = speed_profile_kwargs or {}
        if load_initializer is not None:
            warnings.warn(
                'Given initializer will be overwritten with starting value '
                'from speed-profile, to avoid complications at the load reset.'
                ' It is recommended to choose starting value of'
                ' load by the defined speed-profile.',
                UserWarning)

        self.speed_profile_kwargs = speed_profile_kwargs
        self._speed_profile = speed_profile
        self._tau = tau
        # setting initial load as speed-profile at time 0
        self._omega_initial = self._speed_profile(t=0, **self.speed_profile_kwargs)

    def mechanical_ode(self, t, mechanical_state, torque=None):
        # Docstring of superclass
        # calc next omega with given profile und tau
        omega_next = self._speed_profile(t=t+self._tau, **self.speed_profile_kwargs)
        # calculated T out of euler-forward, given omega_next and
        # actual omega give from system
        return np.array([(1 / self._tau) *
                         (omega_next - mechanical_state[self.OMEGA_IDX])])

    def mechanical_jacobian(self, t, mechanical_state, torque):
        # Docstring of superclass
        # jacobian here not necessary, since omega is externally given
        return None

    def reset(self, **kwargs):
        # Docstring of superclass
        return np.asarray(self._omega_initial, dtype=float)[None]
