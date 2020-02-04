import pytest
from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad, MechanicalLoad
import numpy as np
import math

# The load parameter values used in the test
load_parameter1 = {'j_load': 0.2, 'state_names': ['omega'], 'j_rot_load': 0.25, 'omega_range': (0, 1),
                   'parameter': dict(a=0.12, b=0.13, c=0.4, j_load=0.2)}


@pytest.fixture
def defaultMechanicalLoad():
    """
     pytest fixture that returns a default MechanicalLoad object
    :return: MechanicalLoad object initialized with default values
    """
    return MechanicalLoad()


@pytest.fixture
def concreteMechanicalLoad():
    """
     pytest fixture that returns a  MechanicalLoad object with concrete parameters
    :return: MechanicalLoad object initialized with concrete values
    """
    # Parameters picked from the parameter dict above
    state_names = load_parameter1['state_names']
    j_load = load_parameter1['j_load']
    return MechanicalLoad(state_names, j_load, a=2, b=6)


@pytest.fixture
def defaultPolynomialLoad():
    """
    pytest fixture that returns a default PolynomialLoad object
    :return: PolynomialLoad object initialized with default values
    """
    return PolynomialStaticLoad()


@pytest.fixture
def concretePolynomialLoad():
    """
    pytest fixture that returns a  PolynomialLoad object with concrete parameters
    :return: PolynomialLoad object initialized with concrete values
    """
    addnl_params = dict(p=0.5, q=0.98, r=2678.88)
    test_load_params = dict(a=0.01, b=0.05, c=0.1, j_load=0.1, p=0.5, q = 0.98, r= 2678.88)
    return PolynomialStaticLoad(load_parameter=test_load_params, x=3, y=76)  # x, y are random kwargs


def test_InitMechanicalLoad(defaultMechanicalLoad, concreteMechanicalLoad):
    """
    Test whether the MechanicalLoad object attributes are initialized with the correct values
    :param defaultMechanicalLoad:
    :param concreteMechanicalLoad:
    :return:
    """
    assert defaultMechanicalLoad.state_names == load_parameter1['state_names']
    assert defaultMechanicalLoad.j_total == 0
    assert defaultMechanicalLoad.limits == {}
    assert concreteMechanicalLoad.state_names == load_parameter1['state_names']
    assert concreteMechanicalLoad.j_total == load_parameter1['j_load']
    assert concreteMechanicalLoad.limits == {}


def test_MechanicalLoad_set_j_rotor(concreteMechanicalLoad):
    """
    Test the set_j_rotor() function
    :param concreteMechanicalLoad:
    :return:
    """
    test_val = 1
    concreteMechanicalLoad.set_j_rotor(test_val)
    assert concreteMechanicalLoad.j_total == test_val + load_parameter1['j_load']


def test_MechanicalLoad_MechanicalOde(concreteMechanicalLoad):
    """
    test the mechanical_ode() function
    :param concreteMechanicalLoad:
    :return:
    """
    test_args = 0.5, [2, 3, 4], 0.99
    with pytest.raises(NotImplementedError):
        concreteMechanicalLoad.mechanical_ode(*test_args)


def test_MechanicalLoad_reset(concreteMechanicalLoad):
    """
    Test the reset() function
    :param concreteMechanicalLoad:
    :return:
    """
    resetVal = concreteMechanicalLoad.reset(a=7, b=9)  # set additional random kwargs
    testVal = np.zeros_like(load_parameter1['state_names'], dtype=float)
    assert (resetVal == testVal).all()


def test_MechanicalLoad_get_state_space(concreteMechanicalLoad):
    """
    test the get_state_space() function
    :param concreteMechanicalLoad:
    :return:
    """
    test_omega_range = 0, 5
    test_omega_range_2 = 1, 4, 6, 7
    retVal = concreteMechanicalLoad.get_state_space(test_omega_range)
    assert retVal[0]['omega'] == test_omega_range[0]
    assert retVal[1]['omega'] == test_omega_range[1]

    retVal2 = concreteMechanicalLoad.get_state_space(test_omega_range_2)  # additional negative test
    assert retVal2[0]['omega'] == test_omega_range_2[0]
    assert retVal2[1]['omega'] == test_omega_range_2[1]


def test_InitPolynomialStaticLoad(defaultPolynomialLoad):
    """
    Test the __init__() function of PolynomialStaticLoad class with default parameters
    :param defaultPolynomialLoad:
    :return:
    """
    test_default_load_parameterVal = {'a': 0.01, 'b': 0.05, 'c': 0.1, 'j_load': 0.1}
    assert defaultPolynomialLoad.load_parameter == test_default_load_parameterVal


def test_InitPolynomialStaticLoad(concretePolynomialLoad):
    """
    Test the __init__() function of PolynomialStaticLoad class with concrete parameters
    :param concretePolynomialLoad:
    :return:
    """
    test_concrete_load_parameterVal = {'a': 0.01, 'b': 0.05, 'c': 0.1, 'j_load': 0.1, 'p': 0.5, 'q': 0.98,
                                       'r': 2678.88}
    assert concretePolynomialLoad.load_parameter == test_concrete_load_parameterVal


@pytest.mark.parametrize("omega, expected_result", [(-3, 30.6), (0, 20.0), (5, -7.6)])  # to verify all 3 branches
def test_PolynomialStaticLoad_MechanicalOde(concretePolynomialLoad, omega, expected_result):
    """
    test the mechanical_ode() function of the PolynomialStaticLoad class
    :param concretePolynomialLoad:
    :param omega:
    :return:
    """

    test_mechanical_state = np.array([omega])
    test_t = 1
    test_torque = 2
    op = PolynomialStaticLoad()
    output_val = op.mechanical_ode(test_t, test_mechanical_state, test_torque)
    # output_val = concretePolynomialLoad.mechanical_ode(test_t, test_mechanical_state, test_torque)
    assert math.isclose(expected_result, output_val, abs_tol=1E-6)
