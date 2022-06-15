import gym
import numpy as np

import gym_electric_motor as gem
from gym_electric_motor.physical_system_wrappers import PhysicalSystemWrapper


class FluxObserver(PhysicalSystemWrapper):
    """The FluxObserver extends the systems state vector of induction machine environments by estimated flux states
    ``psi_abs``, and ``psi_angle``. The flux is estimated as follows:

        .. math:: psi_{abs} =  |\Psi|

        .. math:: psi_{angle} = \\angle{\Psi}

        .. math:: \Psi \in \mathbb{C}

        .. math:: I_{s, \\alpha \\beta} = \\left( I_{s,\\alpha}, I_{s, \\beta} \\right) ^T

        .. math::
            \Delta \Psi_k = \\frac {(I_{s, \\alpha}+jI_{s, \\beta}) R_r L_m}{L_r}
            - \Psi_{k-1}(\\frac{R_r}{L_r}+ j\omega)

        .. math ::
            \Psi_k = \sum_{i=0}^k (\Psi_{k-1} + \Delta\Psi_k) \\tau
    """

    def __init__(self, current_names=('i_sa', 'i_sb', 'i_sc'), physical_system=None):
        """
        Args:
            current_names(Iterable[string]): Names of the currents to be observed to estimate the flux.
                (Default: ``('i_sa', 'i_sb', 'i_sc')``)
            physical_system(PhysicalSystem): (Optional) Physical System to initialize this observer. If not passed,
                the observer will be initialized during environment creation.
        """
        self._current_indices = None
        self._l_m = None  # Main induction
        self._l_r = None  # Induction of the rotor
        self._r_r = None  # Rotor resistance
        self._p = None  # Pole pair number
        self._i_s_idx = None
        self._omega_idx = None

        # Integrated values of the flux for the two directions (Re: alpha, Im: beta)
        self._integrated = np.complex(0, 0)
        self._current_names = current_names
        super(FluxObserver, self).__init__(physical_system)

    @staticmethod
    def _abc_to_alphabeta_transformation(i_s):
        return gem.physical_systems.electric_motors.ThreePhaseMotor.t_23(i_s)

    def set_physical_system(self, physical_system):
        # Docstring of super class
        assert isinstance(physical_system.electrical_motor, gem.physical_systems.electric_motors.InductionMotor)
        super().set_physical_system(physical_system)
        mp = physical_system.electrical_motor.motor_parameter
        self._l_m = mp['l_m']  # Main induction
        self._l_r = mp['l_m'] + mp['l_sigr']  # Induction of the rotor
        self._r_r = mp['r_r']  # Rotor resistance
        self._p = mp['p']  # Pole pair number
        psi_limit = self._l_m * physical_system.limits[physical_system.state_names.index('i_sd')]
        low = np.concatenate((physical_system.state_space.low, [-psi_limit, -np.pi]))
        high = np.concatenate((physical_system.state_space.high, [psi_limit, np.pi]))
        self.state_space = gym.spaces.Box(low, high, dtype=np.float64)
        self._current_indices = [physical_system.state_positions[name] for name in self._current_names]
        self._limits = np.concatenate((physical_system.limits, [psi_limit, np.pi]))
        self._nominal_state = np.concatenate((physical_system.nominal_state, [psi_limit, np.pi]))
        self._state_names = physical_system.state_names + ['psi_abs', 'psi_angle']
        self._state_positions = {key: index for index, key in enumerate(self._state_names)}

        self._i_s_idx = [physical_system.state_positions[name] for name in self._current_names]
        self._omega_idx = physical_system.state_positions['omega']
        return self

    def reset(self):
        # Docstring of super class
        self._integrated = np.complex(0, 0)
        return np.concatenate((super().reset(), [0.0, 0.0]))

    def simulate(self, action):
        # Docstring of super class
        state_norm = self._physical_system.simulate(action)
        state = state_norm * self._physical_system.limits
        i_s = state[self._i_s_idx]
        omega = state[self._omega_idx] * self._p

        # Transform current into alpha, beta coordinates
        [i_s_alpha, i_s_beta] = self._abc_to_alphabeta_transformation(i_s)

        # Calculate delta flux
        delta_psi = np.complex(i_s_alpha, i_s_beta) * self._r_r * self._l_m / self._l_r \
            - self._integrated * np.complex(self._r_r / self._l_r, -omega)

        # Integrate the flux
        self._integrated += delta_psi * self._physical_system.tau
        return np.concatenate((state, [np.abs(self._integrated), np.angle(self._integrated)])) / self._limits
