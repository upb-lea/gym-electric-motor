import numpy as np

from .dc_motor import DcMotor


class DcExternallyExcitedMotor(DcMotor):
    # Equals DC Base Motor
    HAS_JACOBIAN = True

    def electrical_jacobian(self, state, u_in, omega, *_):
        mp = self._motor_parameter
        return (
            np.array([
                [-mp['r_a'] / mp['l_a'], -mp['l_e_prime'] / mp['l_a'] * omega],
                [0, -mp['r_e'] / mp['l_e']]
            ]),
            np.array([-mp['l_e_prime'] * state[self.I_E_IDX] / mp['l_a'], 0]),
            np.array([mp['l_e_prime'] * state[self.I_E_IDX],
                      mp['l_e_prime'] * state[self.I_A_IDX]])
        )

    def _update_limits(self):
        # Docstring of superclass

        # R_a might be 0, protect against that
        r_a = 1 if self._motor_parameter['r_a'] == 0 else self._motor_parameter['r_a']

        limit_agenda = \
            {'u_a': self._default_limits['u'],
             'u_e': self._default_limits['u'],
             'i_a': self._limits.get('i', None) or
                    self._limits['u'] / r_a,
             'i_e': self._limits.get('i', None) or
                    self._limits['u'] / self.motor_parameter['r_e'],
             }
        super()._update_limits(limit_agenda)
