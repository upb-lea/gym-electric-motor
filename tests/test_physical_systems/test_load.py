from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad, MechanicalLoad
from ..conf import *
import numpy as np
import pytest


# region first version tests

def test_mechanical_load():
    """
    test mechanical load for different use cases
    :return:
    """
    state_names = load_parameter['state_names']
    # example for one random motor
    state_positions = permex_state_positions
    nominal_state = permex_motor_parameter['nominal_values']
    j_load = load_parameter['j_load']
    j_rotor = load_parameter['j_rot_load']
    omega_range = load_parameter['omega_range']
    # initialize loads
    load_default = MechanicalLoad()
    load_init = MechanicalLoad(state_names, j_load)
    loads = [load_default, load_init]
    # test different initializations of loads
    for load in loads:
        load.set_j_rotor(j_rotor)
        # test state space
        state_space = load.get_state_space(omega_range)
        reset_state_space = Box(low=0, high=1.0, shape=(1,))
        assert state_space[0]['omega'] == omega_range[0]
        assert state_space[1]['omega'] == omega_range[1]
        # test reset
        assert 0 == load.reset(reset_state_space, state_positions,
                               nominal_state)
        # test not existing mechanical ode
        with pytest.raises(NotImplementedError):
            load.mechanical_ode(0, 0, 0)


def test_polynomial_load():
    """
    test mechanical load for different use cases
    different initializations, also wrong ones
    different angular velocities
    :return:
    """

    state_names = load_parameter['state_names']
    # example for one random motor
    state_positions = permex_state_positions
    nominal_state = permex_motor_parameter['nominal_values']
    j_rotor = load_parameter['j_rot_load']
    omega_range = load_parameter['omega_range']
    # initialization loads in different ways
    load_default = PolynomialStaticLoad()
    load_init = PolynomialStaticLoad(load_parameter=load_parameter['parameter'])
    # test initialization of parametrized load function
    assert load_init.load_parameter == load_parameter['parameter'], "Wrong Parameter " + str(load_init.load_parameter) + \
                                                                    str(load_parameter['parameter'])
    assert load_init._a == load_parameter['parameter']['a']
    assert load_init._b == load_parameter['parameter']['b']
    assert load_init._c == load_parameter['parameter']['c']
    # test different loads
    loads = [load_default, load_init]
    for load in loads:
        load.set_j_rotor(j_rotor)
        state_space = load.get_state_space(omega_range)
        reset_state_space = Box(low=0, high=1.0, shape=(len(state_names),))
        assert 0 == load.reset(reset_state_space, state_positions, nominal_state)
        assert state_space[0]['omega'] == omega_range[0]
        assert state_space[1]['omega'] == omega_range[1]
        a = load._a
        b = load._b
        c = load._c
        j_total = load.j_total
        for omega in [-10, 0, 10]:
            electrical_torque = np.sign(omega) * (c * omega**2 + b * abs(omega) + a)
            assert load._static_load(omega) == electrical_torque
            mechanical_state = np.array([omega])
            # test load ode
            for torque in [-3, 0, 5]:
                assert load.mechanical_ode(0, mechanical_state, torque) == np.array([(torque - electrical_torque) /
                                                                                     j_total])


# endregion
