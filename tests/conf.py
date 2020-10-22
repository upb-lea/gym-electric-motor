import math
from matplotlib import pyplot as plt
import matplotlib
from gym_electric_motor.core import *
from gym.spaces import Box
import pytest

# region parameter definition


u_sup = 450.0
series_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                              'j_rotor': 0.017, 'u_sup': u_sup},
                          'limit_values': {'omega': 400, 'torque': 50, 'i': 75, 'u': 430},
                          'nominal_values': {'omega': 370, 'torque': 40, 'i': 50, 'u': 430},
                          'reward_weights': {'omega': 1, 'torque': 0, 'i': 0, 'u': 0, 'u_sup': 0}}
series_state_positions = {'omega': 0, 'torque': 1, 'i': 2, 'u': 3, 'u_sup': 4}
series_state_space = Box(low=-1, high=1, shape=(5,))
series_initializer = {'states': {'i': 0.0},
                      'interval': None,
                      'random_init': None,
                      'random_params': (None, None)}

shunt_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                             'j_rotor': 0.017, 'u_sup': u_sup},
                         'limit_values': {'omega': 400, 'torque': 50, 'i_a': 75, 'i_e': 15, 'u': 430},
                         'nominal_values': {'omega': 368, 'torque': 40, 'i_a': 50, 'i_e': 5, 'u': 430},
                         'reward_weights': {'omega': 1, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u': 0, 'u_sup': 0}}
shunt_state_positions = {'omega': 0, 'torque': 1, 'i_a': 2, 'i_e': 3, 'u': 4, 'u_sup': 5}
shunt_state_space = Box(low=-1, high=1, shape=(6,))
shunt_initializer = {'states': {'i_a': 0.0, 'i_e': 0.0},
                     'interval': None,
                     'random_init': None,
                     'random_params': (None, None)}

extex_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'r_e': 35, 'l_a': 6.3e-3, 'l_e': 160e-3, 'l_e_prime': 0.95,
                                             'j_rotor': 0.017, 'u_sup': u_sup},
                         'limit_values': {'omega': 400, 'torque': 50, 'i_a': 75, 'i_e': 15, 'u_a': 460, 'u_e': 460,
                                          'u': 460},
                         'nominal_values': {'omega': 368, 'torque': 40, 'i_a': 50, 'i_e': 20, 'u_a': 460, 'u_e': 460,
                                            'u': 460},
                         'reward_weights': {'omega': 1, 'torque': 0, 'i_a': 0, 'i_e': 0, 'u_a': 0, 'u_e': 0,
                                            'u_sup': 0}}
extex_state_positions = {'omega': 0, 'torque': 1, 'i_a': 2, 'i_e': 3, 'u_a': 4, 'u_e': 5, 'u_sup': 6}
extex_state_space = Box(low=-1, high=1, shape=(7,))
extex_initializer = {'states': {'i_a': 0.0, 'i_e': 0.0},
                        'interval': None,
                        'random_init': None,
                        'random_params': (None, None)}


permex_motor_parameter = {'motor_parameter': {'r_a': 3.78, 'l_a': 6.3e-3, 'psi_e': 160e-3, 'j_rotor': 0.017,
                                              'u_sup': u_sup},
                          'limit_values': {'omega': 400, 'torque': 50, 'i': 75, 'u': 460},
                          'nominal_values': {'omega': 368, 'torque': 40, 'i': 50, 'u': 460},
                          'reward_weights': {'omega': 1, 'torque': 0, 'i': 0, 'u': 0, 'u_sup': 0}}
permex_state_positions = {'omega': 0, 'torque': 1, 'i': 2, 'u': 3, 'u_sup': 4}
permex_state_space = Box(low=-1, high=1, shape=(5,))
permex_initializer = {'states': {'i': 0.0},
                      'interval': None,
                      'random_init': None,
                      'random_params': (None, None)}

pmsm_motor_parameter = {'motor_parameter': {'p': 3, 'l_d': 84e-3, 'l_q': 125e-3, 'j_rotor': 2.61e-3, 'r_s': 5.0,
                                            'psi_p': 0.171, 'u_sup': u_sup},
                        'limit_values': dict(omega=75, torque=65, i=25, epsilon=math.pi, u=450),
                        'nominal_values': dict(omega=65, torque=50, i=20, epsilon=math.pi, u=450),
                        'reward_weights': dict(omega=1, torque=0, i_a=0, i_b=0, i_c=0, u_a=0, u_b=0, u_c=0, epsilon=0,
                                               u_sup=0)}
pmsm_state_positions = {'omega': 0, 'torque': 1, 'i_a': 2, 'i_b': 3, 'i_c': 4,
                        'i_sq': 5, 'i_sd': 6, 'u_a': 7, 'u_b': 8, 'u_c': 9,
                        'u_sq': 10, 'u_sd': 11, 'epsilon': 12, 'u_sup': 13}
pmsm_state_space = Box(low=-1, high=1, shape=(14,))
pmsm_initializer = {'states': {'i_sq': 0.0, 'i_sd': 0.0, 'epsilon': 0.0},
                    'interval': None,
                    'random_init': None,
                    'random_params': (None, None)}

synrm_motor_parameter = {
    'motor_parameter': {'p': 3, 'l_d': 70e-3, 'l_q': 8e-3, 'j_rotor': 3e-3, 'r_s': 0.5, 'u_sup': u_sup},
    'nominal_values': {'i': 60, 'torque': 65, 'omega': 450.0, 'epsilon': np.pi, 'u': 450},
    'limit_values': {'i': 75, 'torque': 75, 'omega': 550.0, 'epsilon': np.pi, 'u': 450},
    'reward_weights': dict(omega=1, torque=0, i_a=0, i_b=0, i_c=0, u_a=0, u_b=0, u_c=0, epsilon=0, u_sup=0)}
synrm_state_positions = {'omega': 0, 'torque': 1, 'i_a': 2, 'i_b': 3, 'i_c': 4,
                         'i_sq': 5, 'i_sd': 6, 'u_a': 7, 'u_b': 8, 'u_c': 9,
                         'u_sq': 10, 'u_sd': 11, 'epsilon': 12, 'u_sup': 13}
synrm_state_space = Box(low=-1, high=1, shape=(14,))
synrm_initializer = {'states': {'i_sq': 0.0, 'i_sd': 0.0, 'epsilon': 0.0},
                     'interval': None,
                     'random_init': None,
                     'random_params': (None, None)}

sci_motor_parameter = {
    'motor_parameter': {'p': 2, 'l_m': 140e-3, 'l_sigs': 5e-3, 'l_sigr': 5e-3, 'j_rotor': 0.001, 'r_s': 3, 'r_r': 1.5, 'u_sup': u_sup},
    'nominal_values': {'i': 3.9, 'torque': 4.7, 'omega': 314., 'epsilon': np.pi, 'u': 560},
    'limit_values': {'i': 5.5, 'torque': 6, 'omega': 350.0, 'epsilon': np.pi, 'u': 560},
    'reward_weights': dict(omega=1, torque=0, i_sa=0, i_sb=0, i_sc=0, u_sa=0, u_sb=0, u_sc=0, epsilon=0, u_sup=0)}
sci_state_positions = {'omega': 0, 'torque': 1, 'i_sa': 2, 'i_sb': 3, 'i_sc': 4,
                        'i_sq': 5, 'i_sd': 6, 'u_sa': 7, 'u_sb': 8, 'u_sc': 9,
                        'u_sq': 10, 'u_sd': 11, 'epsilon': 12, 'u_sup': 13}
sci_state_space = Box(low=-1, high=1, shape=(14,))
sci_initializer = {'states': {'i_salpha': 0.0, 'i_sbeta': 0.0,
                               'psi_ralpha': 0.0, 'psi_rbeta': 0.0,
                               'epsilon': 0.0},
                   'interval': None,
                   'random_init': None,
                   'random_params': (None, None)}

dfim_state_positions = {'omega': 0, 'torque': 1, 'i_sa': 2, 'i_sb': 3, 'i_sc': 4,
                        'i_sq': 5, 'i_sd': 6, 'i_ra': 7, 'i_rb': 8, 'i_rc': 9,
                        'i_rq': 10, 'i_rd': 11, 'u_sa': 12, 'u_sb': 13, 'u_sc': 14,
                        'u_sq': 15, 'u_sd': 16, 'u_ra': 17, 'u_rb': 18, 'u_rc': 19,
                        'u_rq': 20, 'u_rd': 21, 'epsilon': 22, 'u_sup': 23}
dfim_state_space = Box(low=-1, high=1, shape=(24,))
dfim_initializer = {'states': {'i_salpha': 0.0, 'i_sbeta': 0.0,
                                'psi_ralpha': 0.0, 'psi_rbeta': 0.0,
                                'epsilon': 0.0},
                    'interval': None,
                    'random_init': None,
                    'random_params': (None, None)}

load_parameter = {'j_load': 0.2, 'state_names': ['omega'], 'j_rot_load': 0.25, 'omega_range': (0, 1),
                  'parameter': dict(a=0.12, b=0.13, c=0.4, j_load=0.2)}

converter_parameter = {'tau': 2E-4, 'dead_time': True, 'interlocking_time': 1E-6}

test_motor_parameter = {'DcSeries': series_motor_parameter,
                        'DcShunt': shunt_motor_parameter,
                        'DcPermEx': permex_motor_parameter,
                        'DcExtEx': extex_motor_parameter,
                        'PMSM': pmsm_motor_parameter,
                        'SynRM': synrm_motor_parameter,
                        'SCIM': sci_motor_parameter,
                        'DFIM': sci_motor_parameter,}

test_motor_initializer = {'DcSeries': series_initializer,
                          'DcShunt': shunt_initializer,
                          'DcPermEx': permex_initializer,
                          'DcExtEx': extex_initializer,
                          'PMSM': pmsm_initializer,
                          'SynRM': synrm_initializer,
                          'SCIM': sci_initializer,
                          'DFIM': dfim_initializer,}


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

