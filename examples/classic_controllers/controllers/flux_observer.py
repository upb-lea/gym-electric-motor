import numpy as np


class FluxObserver:
    """
    This class represents a rotor flux observer for an induction motor base on a current model. Further information
    can be found at https://ieeexplore.ieee.org/document/4270863.
    """

    def __init__(self, env):
        mp = env.physical_system.electrical_motor.motor_parameter
        self.l_m = mp["l_m"]  # Main induction
        self.l_r = mp["l_m"] + mp["l_sigr"]  # Induction of the rotor
        self.r_r = mp["r_r"]  # Rotor resistance
        self.p = mp["p"]  # Pole pair number
        self.tau = env.physical_system.tau  # Sampling time

        # function to transform the currents from abc to alpha/beta coordinates
        self.abc_to_alphabeta_transformation = (
            env.physical_system.abc_to_alphabeta_space
        )

        # Integrated values of the flux for the two directions (Re: alpha, Im: beta)
        self.integrated = np.complex(0, 0)
        self.i_s_idx = [
            env.state_names.index("i_sa"),
            env.state_names.index("i_sb"),
            env.state_names.index("i_sc"),
        ]
        self.omega_idx = env.state_names.index("omega")

    def estimate(self, state):
        """
        Method to estimate the flux of an induction motor

        Args:
            state: state of the gym-electric-motor environment

        Returns:
            Amount and angle of the estimated flux
        """

        i_s = state[self.i_s_idx]
        omega = state[self.omega_idx] * self.p

        # Transform current into alpha, beta coordinates
        [i_s_alpha, i_s_beta] = self.abc_to_alphabeta_transformation(i_s)

        # Calculate delta flux
        delta = np.complex(
            i_s_alpha, i_s_beta
        ) * self.r_r * self.l_m / self.l_r - self.integrated * np.complex(
            self.r_r / self.l_r, -omega
        )

        # Integrate the flux
        self.integrated += delta * self.tau
        return np.abs(self.integrated), np.angle(self.integrated)

    def reset(self):
        # Reset the integrated value
        self.integrated = np.complex(0, 0)
