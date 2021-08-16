import pytest
import gym_electric_motor as gem
from gym_electric_motor.physical_systems import PolynomialStaticLoad, MechanicalLoad, ConstantSpeedLoad, ExternalSpeedLoad
from gym.spaces import Box
import numpy as np
from scipy import signal
import math

# The load parameter values used in the test
load_parameter1 = {'j_load': 0.2, 'state_names': ['omega'], 'j_rot_load': 0.25, 'omega_range': (0, 1),
                   'parameter': dict(a=0.12, b=0.13, c=0.4, j_load=0.2)}
# The different initializers used in test
test_const_initializer = {'states': {'omega': 15.0},
                          'interval': None,
                          'random_init': None,
                          'random_params': (None, None)}
# todo random init as testcase ?
test_rand_initializer = { 'interval': None,
                          'random_init': None,
                          'random_params': (None, None)}
# profile_parameter
test_amp = 20
test_bias = 10
test_freq = 2


def speed_profile_(t, amp, freq, bias):
    return amp*signal.sawtooth(2*np.pi*freq*t, width=0.5)+bias


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
    test_initializer = test_const_initializer
    return MechanicalLoad(state_names, j_load, load_initializer=test_initializer)


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
    test_load_params = dict(a=0.01, b=0.05, c=0.1, j_load=0.1)
    test_initializer = test_const_initializer
    # x, y are random kwargs
    return PolynomialStaticLoad(load_parameter=test_load_params, load_initializer=test_initializer)


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
    assert defaultMechanicalLoad.initializer == {}
    assert concreteMechanicalLoad.state_names == load_parameter1['state_names']
    assert concreteMechanicalLoad.j_total == load_parameter1['j_load']
    assert concreteMechanicalLoad.limits == {}
    assert concreteMechanicalLoad.initializer == test_const_initializer


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
    # random testcase for the necessary parameters needed for initialization
    test_positions = {'omega': 0}
    test_nominal = np.array([80])
    # gym.Box state space with random size
    test_space = Box(low=-1.0, high=1.0, shape=(3,))
    # set additional random kwargs
    resetVal = concreteMechanicalLoad.reset(a=7, b=9,
                                            state_positions=test_positions,
                                            nominal_state=test_nominal,
                                            state_space=test_space
                                            )
    testVal = np.asarray(list(test_const_initializer['states'].values()))
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
    test_concrete_load_parameterVal = {'a': 0.01, 'b': 0.05, 'c': 0.1, 'j_load': 0.1}
    assert concretePolynomialLoad.load_parameter == test_concrete_load_parameterVal


@pytest.mark.parametrize("omega, expected_result", [(-3, 23400), (0, 20000), (5, 11400)])  # to verify all 3 branches
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
    load_parameter = dict(j_load=1e-4, a=0.01, b=0.02, c=0.03)
    op = PolynomialStaticLoad(load_parameter=load_parameter)
    output_val = op.mechanical_ode(test_t, test_mechanical_state, test_torque)
    # output_val = concretePolynomialLoad.mechanical_ode(test_t, test_mechanical_state, test_torque)
    assert math.isclose(expected_result, output_val, abs_tol=1E-6)


class TestMechanicalLoad:
    key = ''
    class_to_test = MechanicalLoad
    kwargs = {}

    def test_registered(self):
        if self.key != '':
            load = gem.utils.instantiate(MechanicalLoad, self.key, **self.kwargs)
            assert type(load) == self.class_to_test


class TestConstSpeedLoad(TestMechanicalLoad):

    key = 'ConstSpeedLoad'
    class_to_test = ConstantSpeedLoad

    @pytest.fixture
    def const_speed_load(self):
        return ConstantSpeedLoad(omega_fixed=60)

    def test_initialization(self):
        load = ConstantSpeedLoad(omega_fixed=60)
        assert load.omega_fixed == 60

    def test_mechanical_ode(self, const_speed_load):
        assert all(const_speed_load.mechanical_ode() == np.array([0]))

    def test_reset(self, const_speed_load):
        test_positions = {'omega': 0}
        test_nominal = np.array([80])
        # gym.Box state space with random size
        test_space = Box(low=-1.0, high=1.0, shape=(3,))
        reset_val = const_speed_load.reset(state_positions=test_positions,
                                           nominal_state=test_nominal,
                                           state_space=test_space)
        # set additional random kwargs
        assert all(reset_val == np.array([const_speed_load.omega_fixed]))

    @pytest.mark.parametrize('omega, omega_fixed, expected', [
        (-0.5, 1000, (0, 0)),
        (0, 0, (0, 0)),
        (2, -523, (0, 0)),
        (2, 0.2, (0, 0))
    ]
    )
    def test_jacobian(self, omega, omega_fixed, expected):
        test_object = self.class_to_test(omega_fixed)

        # 2 Runs to test independence on time and torque
        result0 = test_object.mechanical_jacobian(0.456, np.array([omega]), 0.385)
        result1 = test_object.mechanical_jacobian(5.345, np.array([omega]), -0.654)

        assert result0[0] == result1[0] == expected[0]
        assert result0[1] == result1[1] == expected[1]


class TestExtSpeedLoad(TestMechanicalLoad):

    key = 'ExtSpeedLoad'
    class_to_test = ExternalSpeedLoad
    kwargs = dict(
        speed_profile=speed_profile_,
        speed_profile_kwargs=dict(
            amp=test_amp,
            bias=test_bias,
            freq=test_freq
        )
    )

    @pytest.fixture
    def ext_speed_load(self):
        return ExternalSpeedLoad(
            speed_profile=speed_profile_,
            speed_profile_kwargs=dict(amp=test_amp, bias=test_bias, freq=test_freq)
        )

    def test_initialization(self):
        load = ExternalSpeedLoad(
            speed_profile=speed_profile_,
            speed_profile_kwargs=dict(amp=test_amp, bias=test_bias, freq=test_freq)
        )
        assert load._speed_profile == speed_profile_
        assert load.omega == speed_profile_(t=0, amp=test_amp, bias=test_bias, freq=test_freq)
        for key in ['amp', 'bias', 'freq']:
            assert key in load.speed_profile_kwargs

    # to verify all 3 branches
    @pytest.mark.parametrize("omega, expected_result", [(-3, -69840.),
                                                        (0, -99840.),
                                                        (5, -149840.)])
    def test_mechanical_ode(self, ext_speed_load, omega, expected_result):
        test_mechanical_state = np.array([omega])
        test_t = 1
        op = ext_speed_load
        output_val = op.mechanical_ode(test_t, test_mechanical_state)
        assert math.isclose(expected_result, output_val, abs_tol=1E-6)

    def test_reset(self, ext_speed_load):
        test_positions = {'omega': 0}
        test_nominal = np.array([80])
        # gym.Box state space with random size
        test_space = Box(low=-1.0, high=1.0, shape=(3,))
        reset_var = ext_speed_load.reset(state_positions=test_positions,
                                         nominal_state=test_nominal,
                                         state_space=test_space)
        assert all(reset_var == np.array([ext_speed_load._omega_initial]))

    @pytest.mark.parametrize('omega, omega_initial, expected', [
        (-0.5, 1000, (None, None)),
        (0, 0, (None, None)),
        (2, -523, (None, None)),
        (2, 0.2, (None, None))
    ]
    )
    def test_jacobian(self, omega, omega_initial, expected):
        test_object = self.class_to_test(
            speed_profile_, speed_profile_kwargs=dict(amp=test_amp, bias=test_bias, freq=test_freq)
        )

        # 2 Runs to test independence on time and torque
        result0 = test_object.mechanical_jacobian(0.456, np.array([omega]), 0.385)
        result1 = test_object.mechanical_jacobian(5.345, np.array([omega]), -0.654)

        assert result0 == result1 == expected[0]
        assert result0 == result1 == expected[1]


class TestPolyStaticLoad(TestMechanicalLoad):

    class_to_test = PolynomialStaticLoad
    key = 'PolyStaticLoad'

    @pytest.mark.parametrize('omega, load_parameter, expected', [
        (-0.5, dict(a=12, b=1, c=0, j_load=1), (1, 1)),
        (0, dict(j_load=0.5), (0, 2)),
        (2, dict(a=20, b=0, c=2, j_load=0.25), (-32, 4)),
        (2, dict(a=20, b=0.125, c=2, j_load=0.25), (-32.5, 4))
    ]
    )
    def test_jacobian(self, omega, load_parameter, expected):
        test_object = self.class_to_test(load_parameter)

        # 2 Runs to test independence on time and torque
        result0 = test_object.mechanical_jacobian(0.456, np.array([omega]), 0.385)
        result1 = test_object.mechanical_jacobian(5.345, np.array([omega]), -0.654)

        assert result0[0] == result1[0] == expected[0]
        assert result0[1] == result1[1] == expected[1]
