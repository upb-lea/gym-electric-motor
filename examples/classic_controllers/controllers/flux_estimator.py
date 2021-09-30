import numpy as np


class FluxEstimator:
    """
        This class represents a rotor flux observer for an induction motor base on a current model. Further information
        can be found at https://ieeexplore.ieee.org/document/4270863.
    """
    def __init__(self, env):
        mp = env.physical_system.electrical_motor.motor_parameter
        self.l_m = mp['l_m']
        self.l_r = mp['l_m'] + mp['l_sigr']
        self.r_r = mp['r_r']
        self.T23 = env.physical_system.electrical_motor.t_23
        self.abc_to_alphabeta_transformation = env.physical_system.abc_to_alphabeta_space
        self.tau = env.physical_system.tau
        self.integrated = np.complex(0, 0)
        self.i_s_idx = [env.state_names.index('i_sa'), env.state_names.index('i_sb'), env.state_names.index('i_sc')]
        self.omega_idx = env.state_names.index('omega')
        self.i_limit = env.physical_system.limits[env.state_names.index('i_sa')]
        self.o_limit = env.physical_system.limits[env.state_names.index('omega')]
        self.p = mp['p']

    def estimate(self, state):
        i_s = state[self.i_s_idx]
        omega = state[self.omega_idx] * self.p
        [i_s_alpha, i_s_beta] = self.abc_to_alphabeta_transformation(i_s)
        delta = np.complex(i_s_alpha, i_s_beta) * self.r_r * self.l_m / self.l_r - self.integrated * np.complex(
            self.r_r / self.l_r, -omega)
        self.integrated += delta * self.tau
        return np.abs(self.integrated), np.angle(self.integrated)

    def reset(self):
        self.integrated = 0
