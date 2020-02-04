import math
from matplotlib import pyplot as plt
import matplotlib
from PyQt5.QtWidgets import QWidget
from gym_electric_motor.core import *

import pytest

# region parameter definition


u_sup = 450.0
series_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                              'j_rotor': 0.017, 'u_sup': u_sup},
                          'limit_values': {'omega': 400, 'torque': 50, 'i': 75, 'u': 430},
                          'nominal_values': {'omega': 370, 'torque': 40, 'i': 50, 'u': 430},
                          'reward_weights': {'omega': 1, 'torque': 0, 'i': 0, 'u': 0, 'u_sup': 0}}

shunt_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                             'j_rotor': 0.017, 'u_sup': u_sup},
                         'limit_values': {'omega': 400, 'torque': 50, 'i_a': 75, 'i_e': 15, 'u': 430},
                         'nominal_values': {'omega': 368, 'torque': 40, 'i_a': 50, 'i_e': 5, 'u': 430},
                         'reward_weights': {'omega': 1, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u': 0, 'u_sup': 0}}

extex_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                             'j_rotor': 0.017, 'u_sup': u_sup},
                         'limit_values': {'omega': 400, 'torque': 50, 'i_a': 75, 'i_e': 15, 'u_a': 460, 'u_e': 460,
                                          'u': 460},
                         'nominal_values': {'omega': 368, 'torque': 40, 'i_a': 50, 'i_e': 20, 'u_a': 460, 'u_e': 460,
                                            'u': 460},
                         'reward_weights': {'omega': 1, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u_a': 0, 'u_e': 0,
                                            'u_sup': 0}}

permex_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'l_a': 6.3e-3, 'psi_e': 160e-3, 'j_rotor': 0.017,
                                              'u_sup': u_sup},
                          'limit_values': {'omega': 400, 'torque': 50, 'i': 75, 'u': 460},
                          'nominal_values': {'omega': 368, 'torque': 40, 'i': 50, 'u': 460},
                          'reward_weights': {'omega': 1, 'torque': 0, 'i': 0, 'u': 0, 'u_sup': 0}}

pmsm_motor_parameter = {'motor_parameter': {'p': 3, 'l_d': 84e-3, 'l_q': 125e-3, 'j_rotor': 2.61e-3, 'r_s': 5.0,
                                            'psi_p': 0.171, 'u_sup': u_sup},
                        'limit_values': dict(omega=75, torque=65, i=25, epsilon=math.pi, u=450),
                        'nominal_values': dict(omega=65, torque=50, i=20, epsilon=math.pi, u=450),
                        'reward_weights': dict(omega=1, torque=0, i_a=0, i_b=0, i_c=0, u_a=0, u_b=0, u_c=0, epsilon=0,
                                               u_sup=0)}
synrm_motor_parameter = {
    'motor_parameter': {'p': 3, 'l_d': 70e-3, 'l_q': 8e-3, 'j_rotor': 3e-3, 'r_s': 0.5, 'u_sup': u_sup},
    'nominal_values': {'i': 60, 'torque': 65, 'omega': 450.0, 'epsilon': np.pi, 'u': 450},
    'limit_values': {'i': 75, 'torque': 75, 'omega': 550.0, 'epsilon': np.pi, 'u': 450},
    'reward_weights': dict(omega=1, torque=0, i_a=0, i_b=0, i_c=0, u_a=0, u_b=0, u_c=0, epsilon=0, u_sup=0)}

load_parameter = {'j_load': 0.2, 'state_names': ['omega'], 'j_rot_load': 0.25, 'omega_range': (0, 1),
                  'parameter': dict(a=0.12, b=0.13, c=0.4, j_load=0.2)}

converter_parameter = {'tau': 2E-4, 'dead_time': True, 'interlocking_time': 1E-6}

test_motor_parameter = {'DcSeries': series_motor_parameter,
                        'DcShunt': shunt_motor_parameter,
                        'DcPermEx': permex_motor_parameter,
                        'DcExtEx': extex_motor_parameter,
                        'PMSM': pmsm_motor_parameter,
                        'SynRM': synrm_motor_parameter}


# endregion

# region render window turn off
# use the following function, that no window is shown while testing if env.render() is called

def monkey_ion_function():
    """
    function used for plt.ion()
    :return:
    """
    pass


def monkey_pause_function(time=None):
    """
    function used instead of plt.pause()
    :param time:
    :return:
    """
    pass


def monkey_show_maximized_function(args=None):
    """
    function used instead of figureManager.showMaximized()
    :param args:
    :return:
    """
    pass


def monkey_show_function(args=None):
    """
    function used instead of self._figure.show()
    :param args:
    :return:
    """
    pass


@pytest.fixture(scope='function')
def turn_off_windows(monkeypatch):
    """
    This preparation function is run before each test. Due to this, no rendering is performed.
    :param monkeypatch:
    :return:
    """
    monkeypatch.setattr(plt, "ion", monkey_ion_function)
    monkeypatch.setattr(plt, "pause", monkey_pause_function)
    monkeypatch.setattr(QWidget, "showMaximized", monkey_show_maximized_function)
    monkeypatch.setattr(matplotlib.figure.Figure, "show", monkey_show_function)


# endregion

# region system equations for testing


"""
    simulate the system
    d/dt[x,y]=[[3 * x + 5 * y - 2 * x * y + 3 * x**2 - 0.5 * y**2],
    [10 - 0.6 * x + 0.9 * y**2 - 3 * x**2 *y + u]]
    with the initial value [1, 6]
"""


def system(t, state, u):
    """
    differential system equation
    :param t: time
    :param state: current state
    :param u: input
    :return: derivative of the current state
    """
    x = state[0]
    y = state[1]
    result = np.array([3 * x + 5 * y - 2 * x * y + 3 * x ** 2 - 0.5 * y ** 2,
                       10 - 0.6 * x + 0.9 * y ** 2 - 3 * x ** 2 * y + u])
    return result


def jacobian(t, state, u):
    """
    jacobian matrix of the differential equation system
    :param t: time
    :param state: current state
    :param u: input
    :return: jacobian matrix
    """
    x = state[0]
    y = state[1]
    result = np.array([[3 - 2 * y + 6 * x, 5 - 2 * x - y], [-0.6 - 6 * x * y, 1.8 * y - 3 * x ** 2]])
    return result


# endregion

