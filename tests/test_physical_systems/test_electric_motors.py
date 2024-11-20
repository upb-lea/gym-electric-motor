import pytest
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import (
    ElectricMotor,
    DcMotor,
    DcExternallyExcitedMotor,
    DcPermanentlyExcitedMotor
)
from gymnasium.spaces import Box
import numpy as np

#parameters for dc motor
test_dcmotor_parameter = {"r_a": 16e-2, "r_e": 16e-1, "l_a": 19e-5, "l_e_prime": 1.7e-2, "l_e": 5.4e-1, "j_rotor": 0.025}
test_dcmotor_nominal_values = dict(omega=350, torque=16.5, i=100, i_a=96, i_e=87, u=65, u_a=65, u_e=61)
test_dcmotor_limits = dict(omega=350, torque=38.0, i=210, i_a=210, i_e=215, u=60, u_a=60, u_e=65)
test_dcmotor_default_initializer = {
        "states": {"i_a": 10.0, "i_e": 15.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }

test_electricmotor_default_initializer = {
        "states": {"omega": 16.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
test_DcPermanentlyExcitedMotor_parameter = {"r_a": 16e-5, "l_a": 19e-2, "psi_e": 0.00165, "j_rotor": 0.05,}
test_DcPermanentlyExcitedMotor_initializer = {"states": {"i": 10.0}, "interval": None, "random_init": None,  "random_params": (None, None),}

@pytest.fixture
def defaultElectricMotor():
    """
     pytest fixture that returns a default electric motor object
    :return: electric motor object initialized with default values
    """
    return ElectricMotor()


@pytest.fixture
def concreteElectricMotor():
    """
     pytest fixture that returns a electric motor object
    :return: Electric motor object initialized with concrete initializer
    """
    return ElectricMotor(None,None,None,test_electricmotor_default_initializer,None)

@pytest.fixture
def defaultDCMotor():
    """
     pytest fixture that returns a DC motor object
    :return: Electric motor object initialized with concrete values
    """
    return DcMotor()

@pytest.fixture
def concreteDCMotor():
    """
     pytest fixture that returns a DC motor object
    :return: Electric motor object initialized with concrete values
    """
    return DcMotor(test_dcmotor_parameter,test_dcmotor_nominal_values,test_dcmotor_limits,test_dcmotor_default_initializer)

@pytest.fixture
def defaultDcExternallyExcitedMotor():
     """
     pytest fixture that returns a DC ExternallyExcited motor object
    :return: ExternallyExcited DC motor object with default values
    """
     return DcExternallyExcitedMotor()

@pytest.fixture
def defaultDcPermanentlyExcitedMotor():
     """
     pytest fixture that returns a Dc Permanently Excited motor object
    :return:  Permanently Excited DC motor object with default values
    """
     return DcPermanentlyExcitedMotor()

@pytest.fixture
def ConcreteDcPermanentlyExcitedMotor():
     """
     pytest fixture that returns a Dc Permanently Excited motor object
    :return:  Permanently Excited DC motor object with concrete values
    """
     return DcPermanentlyExcitedMotor(test_DcPermanentlyExcitedMotor_parameter,None,None,test_DcPermanentlyExcitedMotor_initializer)

def test_InitElectricMotor(defaultElectricMotor,concreteElectricMotor):
    """
    Test whether the Electric motor object attributes are initialized with the correct values
    :param defaultElectricMotor:
    :param concreteElectricMotor:
    :return:
    """
    assert defaultElectricMotor.motor_parameter == {}
    assert defaultElectricMotor.nominal_values == {}
    assert defaultElectricMotor.limits == {}
    assert defaultElectricMotor._initial_states == {}
    assert concreteElectricMotor._initial_states == {'omega': 16.0}
    with pytest.raises(NotImplementedError):
            concreteElectricMotor.electrical_ode(np.random.rand(1, 1),[0,0.5,1,1.5],16.0)

def test_InitDCMotor(defaultDCMotor, concreteDCMotor):
     
     """
     Test whether the DC motor object are initialized with correct values"
     :param defaultDCMotor:
     :param concreteDCMotor:
     :return:
     """
     assert defaultDCMotor.motor_parameter ==  {"r_a": 16e-3, "r_e": 16e-2, "l_a": 19e-6, "l_e_prime": 1.7e-3, "l_e": 5.4e-3, "j_rotor": 0.0025}
     assert defaultDCMotor.nominal_values == dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
     assert concreteDCMotor.motor_parameter == test_dcmotor_parameter
     assert concreteDCMotor.nominal_values == test_dcmotor_nominal_values
     assert concreteDCMotor.limits == test_dcmotor_limits
     assert concreteDCMotor._initial_states == {"i_a": 10.0, "i_e": 15.0}

def test_DCMotor_torque(concreteDCMotor):
     
     currents = [100, 10, 2, 3]
     expectedtorque = concreteDCMotor.motor_parameter["l_e_prime"]* currents[0] * currents[1]

     assert concreteDCMotor.torque(currents) == expectedtorque

def test_DCMotor_i_in(concreteDCMotor):
     
     currents = [100, 10, 2, 3]
     expectedcurrent = currents

     assert concreteDCMotor.i_in(currents) == expectedcurrent

def test_DCMotor_electrical_ode(concreteDCMotor):
     
     concreteDCMotor._model_constants[0] = concreteDCMotor._model_constants[0] / concreteDCMotor.motor_parameter["l_a"]
     concreteDCMotor._model_constants[1] = concreteDCMotor._model_constants[1] / concreteDCMotor.motor_parameter["l_e"]
     u_in = [10,20]
     omega = 60
     state = [10.0,15.0]

     expectedElectricalOde = np.matmul(
            concreteDCMotor._model_constants,
            np.array(
                [
                    state[0],
                    state[1],
                    omega * state[1],
                    u_in[0],
                    u_in[1],
                ]
            ),
        )
     
     assert np.array_equal(concreteDCMotor.electrical_ode(state,u_in,omega), expectedElectricalOde)

def test_DCMotor_get_state_space(concreteDCMotor):
     
    inputvoltages = Box(0, 1, shape=(2,), dtype=np.float64)
    inputscurrents = Box(-1, 1, shape=(2,), dtype=np.float64)

    expectedlow = {
            "omega":0,
            "torque": -1,
            "i_a": -1 ,
            "i_e": -1 ,
            "u_a": 0,
            "u_e":  0,
        }
    
    expectedhigh = {"omega": 1, "torque": 1, "i_a": 1, "i_e": 1, "u_a": 1, "u_e": 1}
    
    assert concreteDCMotor.get_state_space(inputscurrents,inputvoltages)[0] == expectedlow
    assert concreteDCMotor.get_state_space(inputscurrents,inputvoltages)[1] == expectedhigh

def test_ConcreteDCMotor_reset(concreteDCMotor):
    new_initial_state = {'i_a': 100.0, 'i_e': 20.0}
    default_initial_state = {"i_a": 10.0, "i_e": 15.0}
    default_Initial_state_array = [10.0, 15.0]
    extex_state_positions = {
          "omega": 0,
          "torque": 1,
          "i":2,
          "i_a":3,
          "i_e": 4,
          "u_a": 5,
          "u_e": 6,
    }
     
    extex_state_space = Box(low=-1, high=1, shape=(7,), dtype=np.float64)
    assert concreteDCMotor._initial_states == default_initial_state
    concreteDCMotor._initial_states = {'i_a': 100.0, 'i_e': 20.0}

    assert concreteDCMotor._initial_states == new_initial_state
    assert np.array_equal(concreteDCMotor.reset(extex_state_space,extex_state_positions),default_Initial_state_array)
    assert concreteDCMotor._initial_states ==  default_initial_state


def test_defaultDCMotor_reset(defaultDCMotor):
     new_initial_state = {'i_a': 1.0, 'i_e': 2.0}
     default_initial_state = {"i_a": 0.0, "i_e": 0.0}
     default_Initial_state_array = [0.0, 0.0]
     extex_state_positions = {
          "omega": 0,
          "torque": 1,
          "i":2,
          "i_a":3,
          "i_e": 4,
          "u_a": 5,
          "u_e": 6,
      }
     
     extex_state_space = Box(low=-1, high=1, shape=(7,), dtype=np.float64)

     assert defaultDCMotor._initial_states == default_initial_state
     defaultDCMotor._initial_states = {'i_a': 1.0, 'i_e': 2.0}

     assert defaultDCMotor._initial_states == new_initial_state
     assert np.array_equal(defaultDCMotor.reset(extex_state_space,extex_state_positions),default_Initial_state_array)
     assert defaultDCMotor._initial_states ==  default_initial_state

def test_defaultDCExternallyExcitedMotor(defaultDcExternallyExcitedMotor, defaultDCMotor):
     
     assert defaultDcExternallyExcitedMotor.motor_parameter == defaultDCMotor.motor_parameter
     assert defaultDcExternallyExcitedMotor._default_initializer == defaultDCMotor._default_initializer
     assert defaultDcExternallyExcitedMotor._initial_limits == defaultDCMotor._initial_limits
     assert defaultDcExternallyExcitedMotor._nominal_values == defaultDCMotor._nominal_values
    
def test_DcExternallyExcitedMotor_electrical_jacobian(defaultDcExternallyExcitedMotor,defaultDCMotor) :
     mp = defaultDCMotor.motor_parameter
     omega = 60
     state =[0.0,0.0]
     I_A_IDX = 0
     I_E_IDX = 1
     expected_jacobian = (
            np.array(
                [
                    [-mp["r_a"] / mp["l_a"], -mp["l_e_prime"] / mp["l_a"] * omega],
                    [0, -mp["r_e"] / mp["l_e"]],
                ]
            ),
            np.array([-mp["l_e_prime"] * state[I_E_IDX] / mp["l_a"], 0]),
            np.array(
                [
                    mp["l_e_prime"] * state[I_E_IDX],
                    mp["l_e_prime"] * state[I_A_IDX],
                ]
            ),
        )
     assert np.array_equal(defaultDcExternallyExcitedMotor.electrical_jacobian(state,[],omega)[0],expected_jacobian[0])
     assert np.array_equal(defaultDcExternallyExcitedMotor.electrical_jacobian(state,[],omega)[1],expected_jacobian[1])
     assert np.array_equal(defaultDcExternallyExcitedMotor.electrical_jacobian(state,[],omega)[2],expected_jacobian[2])


