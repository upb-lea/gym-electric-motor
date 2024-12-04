import pytest
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import (
    ElectricMotor,
    DcMotor,
    DcExternallyExcitedMotor,
    DcPermanentlyExcitedMotor,
    DcSeriesMotor,
    DcShuntMotor
)
from gymnasium.spaces import Box
import numpy as np

#parameters for dc motor
test_dcmotor_parameter = {"r_a": 16e-2, "r_e": 16e-1, "l_a": 19e-5, "l_e_prime": 1.7e-2, "l_e": 5.4e-1, "j_rotor": 0.025}
test_dcmotor_nominal_values = dict(omega=350, torque=16.5, i=100, i_a=96, i_e=87, u=65, u_a=65, u_e=61)
test_dcmotor_limits = dict(omega=400, torque=38.0, i=210, i_a=210, i_e=215, u=100, u_a=100, u_e=100)
test_dcmotor_default_initializer = {"states": {"i_a": 10.0, "i_e": 15.0}, "interval": None, "random_init": None, "random_params": (None, None),}
test_electricmotor_default_initializer = {"states": {"omega": 16.0}, "interval": None, "random_init": None, "random_params": (None, None),}
test_DcPermanentlyExcitedMotor_parameter = {"r_a": 16e-5, "l_a": 19e-2, "psi_e": 0.00165, "j_rotor": 0.05,}
test_DcPermanentlyExcitedMotor_initializer = {"states": {"i": 10.0}, "interval": None, "random_init": None,  "random_params": (None, None),}
test_DcSeriesMotor_parameter = {"r_a": 16e-2,"r_e": 48e-2,"l_a": 19e-3,"l_e_prime": 1.7e-2,"l_e": 5.4e-2,"j_rotor": 0.25,}
test_DcSeriesMotor_initializer = {"states": {"i": 5.0}, "interval": None, "random_init": None, "random_params": (None, None),}
test_DcShuntMotor_parameter = {"r_a": 16e-3, "r_e": 5e-1, "l_a": 20e-6, "l_e_prime": 1.7e-3, "l_e": 5.4e-3, "j_rotor": 0.025,}
test_DcShuntMotor_initializer = {"states": {"i_a": 10.0, "i_e": 0.0},"interval": None,"random_init": None,"random_params": (None, None),}

@pytest.fixture
def concreteDCMotor():
    """
     pytest fixture that returns a DC motor object
    :return: Electric motor object initialized with concrete values
    """
    return DcMotor(test_dcmotor_parameter,test_dcmotor_nominal_values,test_dcmotor_limits,test_dcmotor_default_initializer)

@pytest.fixture
def concreteDcPermanentlyExcitedMotor():
    """
     pytest fixture that returns a Dc Permanently Excited motor object
    :return:  Permanently Excited DC motor object with concrete values
    """
    return DcPermanentlyExcitedMotor(test_DcPermanentlyExcitedMotor_parameter,None,None,test_DcPermanentlyExcitedMotor_initializer)

@pytest.fixture
def concreteDCSeriesMotor():
    """
     pytest fixture that returns a Dc Series motor object
    :return:  Dc Series motor object with concrete values
    """
    return DcSeriesMotor(test_DcSeriesMotor_parameter,None,None,test_DcSeriesMotor_initializer)

@pytest.fixture
def concreteDCShuntMotor():
    """
     pytest fixture that returns a Dc Shunt motor object
    :return:  Dc Shunt motor object with concrete values
    """
    return DcShuntMotor(test_DcShuntMotor_parameter,None,None,test_DcShuntMotor_initializer)

def test_InitElectricMotor():
    """
    Test whether the Electric motor object attributes are initialized with the correct values

    """
    defaultElectricMotor = ElectricMotor()
    concreteElectricMotor = ElectricMotor(None,None,None,test_electricmotor_default_initializer,None)
    assert defaultElectricMotor.motor_parameter == {}
    assert defaultElectricMotor.nominal_values == {}
    assert defaultElectricMotor.limits == {}
    assert defaultElectricMotor._initial_states == {}
    assert concreteElectricMotor._initial_states == {'omega': 16.0}
    with pytest.raises(NotImplementedError):
            concreteElectricMotor.electrical_ode(np.random.rand(1, 1),[0,0.5,1,1.5],16.0)

def test_InitDCMotor(concreteDCMotor):
     
     """
     Test whether the DC motor object are initialized with correct values"
     :return:
     """
     defaultDCMotor = DcMotor()
     assert defaultDCMotor.motor_parameter ==  {"r_a": 16e-3, "r_e": 16e-2, "l_a": 19e-6, "l_e_prime": 1.7e-3, "l_e": 5.4e-3, "j_rotor": 0.0025}
     assert defaultDCMotor.nominal_values == dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
     assert concreteDCMotor.motor_parameter == test_dcmotor_parameter
     assert concreteDCMotor.nominal_values == test_dcmotor_nominal_values
     assert concreteDCMotor.limits == test_dcmotor_limits
     assert concreteDCMotor._initial_states == test_dcmotor_default_initializer["states"]

def test_DCMotor_torque(concreteDCMotor):
     
     currents = [100, 10, 2, 3]
     expectedtorque = concreteDCMotor.motor_parameter["l_e_prime"]* currents[0] * currents[1]

     assert concreteDCMotor.torque(currents) == expectedtorque

def test_DCMotor_i_in(concreteDCMotor):
     
     currents = [100, 10, 2, 3]
     expectedcurrent = currents

     assert concreteDCMotor.i_in(currents) == expectedcurrent

def test_DCMotor_el_ode(concreteDCMotor):
     
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
     u1 = Box(0, 1, shape=(2,), dtype=np.float64)
     input_current = Box(-1, 1, shape=(2,), dtype=np.float64)
     u2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     expectedLowfor_u1 = {
            "omega":0,
            "torque": -1,
            "i_a": -1 ,
            "i_e": -1 ,
            "u_a": 0,
            "u_e":  0,
        }
     expectedhigh = {"omega": 1, "torque": 1, "i_a": 1, "i_e": 1, "u_a": 1, "u_e": 1}
     expectedLowfor_u2 = {
            "omega":-1,
            "torque": -1,
            "i_a": -1 ,
            "i_e": -1 ,
            "u_a": -1,
            "u_e":  -1,
        }
     assert  expectedLowfor_u1== concreteDCMotor.get_state_space(input_current,u1)[0]
     assert  expectedhigh == concreteDCMotor.get_state_space(input_current,u1)[1]
     assert  expectedLowfor_u2== concreteDCMotor.get_state_space(input_current,u2)[0]
     assert  expectedhigh == concreteDCMotor.get_state_space(input_current,u2)[1]

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
    concreteDCMotor._initial_states = new_initial_state
    assert concreteDCMotor._initial_states == new_initial_state
    assert np.array_equal(concreteDCMotor.reset(extex_state_space,extex_state_positions),default_Initial_state_array)


def test_defaultDCMotor_reset():
     defaultDCMotor = DcMotor()
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
     defaultDCMotor._initial_states = new_initial_state

     assert defaultDCMotor._initial_states == new_initial_state
     assert np.array_equal(defaultDCMotor.reset(extex_state_space,extex_state_positions),default_Initial_state_array)
     assert defaultDCMotor._initial_states ==  default_initial_state

def test_defaultDCExternallyExcitedMotor():
     
     defaultDcExternallyExcitedMotor = DcExternallyExcitedMotor()
     defaultDCMotor = DcMotor()
     assert defaultDcExternallyExcitedMotor.motor_parameter == defaultDCMotor.motor_parameter
     assert defaultDcExternallyExcitedMotor._default_initializer == defaultDCMotor._default_initializer
     assert defaultDcExternallyExcitedMotor._initial_limits == defaultDCMotor._initial_limits
     assert defaultDcExternallyExcitedMotor._nominal_values == defaultDCMotor._nominal_values
"""
No defaultDCExternallyExcitedMotor_reset is tested as it would be same as test_defaultDCMotor_reset
"""

def test_DcExternallyExcitedMotor_el_jacobian() :
     defaultDcExternallyExcitedMotor = DcExternallyExcitedMotor()
     defaultDCMotor = DcMotor()
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

def test_InitDcPermanentlyExcitedMotor(concreteDcPermanentlyExcitedMotor):
     defaultDcPermanentlyExcitedMotor = DcPermanentlyExcitedMotor()
     assert defaultDcPermanentlyExcitedMotor.motor_parameter == {"r_a": 16e-3, "l_a": 19e-6, "psi_e": 0.165, "j_rotor": 0.025,}
     assert defaultDcPermanentlyExcitedMotor._default_initializer == {"states": {"i": 0.0}, "interval": None, "random_init": None,  "random_params": (None, None),}
     assert defaultDcPermanentlyExcitedMotor.nominal_values == dict(omega=300, torque=16.0, i=97, u=60)
     assert defaultDcPermanentlyExcitedMotor.limits == dict(omega=400, torque=38.0, i=210, u=60)
     assert defaultDcPermanentlyExcitedMotor.HAS_JACOBIAN
     assert defaultDcPermanentlyExcitedMotor.CURRENTS_IDX[0] == 0
     assert concreteDcPermanentlyExcitedMotor._default_initializer == defaultDcPermanentlyExcitedMotor._default_initializer
     assert concreteDcPermanentlyExcitedMotor.nominal_values == defaultDcPermanentlyExcitedMotor.nominal_values
     assert concreteDcPermanentlyExcitedMotor.motor_parameter == test_DcPermanentlyExcitedMotor_parameter
     assert concreteDcPermanentlyExcitedMotor.initializer == test_DcPermanentlyExcitedMotor_initializer
     assert concreteDcPermanentlyExcitedMotor._initial_states == test_DcPermanentlyExcitedMotor_initializer["states"]

def test_DcPermanentlyExcitedMotor_torque(concreteDcPermanentlyExcitedMotor):
    state = [100, 10, 2, 3]
    expected_torque = concreteDcPermanentlyExcitedMotor.motor_parameter["psi_e"]*state[0]
    assert concreteDcPermanentlyExcitedMotor.torque(state) == expected_torque

"""
does i_in function will work because of the index [0] instead 0 

def test_DcPermanentlyExcitedMotor_i_in(concreteDcPermanentlyExcitedMotor):
     state = [100,200,300]
     expected_return = state[0]
     assert concreteDcPermanentlyExcitedMotor.i_in(state) == expected_return

"""
def test__DcPermanentlyExcitedMotor_el_ode(concreteDcPermanentlyExcitedMotor):
     state = [10.0,15.0]
     u_in = [10,20]
     omega = 60
     expectedODE = np.matmul(concreteDcPermanentlyExcitedMotor._model_constants, [omega] + np.atleast_1d(state[0]).tolist() + [u_in[0]])
     assert np.array_equal(concreteDcPermanentlyExcitedMotor.electrical_ode(state,u_in,omega),expectedODE)

def test_DcPermanentlyExcitedMotor_el_jacobian(concreteDcPermanentlyExcitedMotor):
     defaultDcPermanentlyExcitedMotor = DcPermanentlyExcitedMotor()
     mp = concreteDcPermanentlyExcitedMotor.motor_parameter
     u_in = [10,20]
     omega = 60
     state = [10.0,15.0]
     expectedJacobian = (
            np.array([[-mp["r_a"] / mp["l_a"]]]),
            np.array([-mp["psi_e"] / mp["l_a"]]),
            np.array([mp["psi_e"]]),
     )
     default_mp = concreteDcPermanentlyExcitedMotor._default_motor_parameter
     default_jacobian = (
            np.array([[-default_mp["r_a"] / default_mp["l_a"]]]),
            np.array([-default_mp["psi_e"] / default_mp["l_a"]]),
            np.array([default_mp["psi_e"]]),
     )
     assert expectedJacobian[0] == concreteDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[0]
     assert expectedJacobian[1] == concreteDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[1]
     assert expectedJacobian[2] == concreteDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[2]
     assert default_jacobian[0] == defaultDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[0]
     assert default_jacobian[1] == defaultDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[1]
     assert default_jacobian[2] == defaultDcPermanentlyExcitedMotor.electrical_jacobian(state,u_in,omega)[2]

def test_DcPermanentlyExcitedMotor_get_state_space(concreteDcPermanentlyExcitedMotor):
     u1 = Box(0, 1, shape=(2,), dtype=np.float64)
     input_current = Box(-1, 1, shape=(2,), dtype=np.float64)
     u2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     expectedLowfor_u1 ={
            "omega": 0,
            "torque": -1,
            "i": -1,
            "u": 0,
        } 
     expectedHighfor_u1 = {
          "omega": 1,
            "torque": 1,
            "i": 1,
            "u": 1,
     }
     expectedLowfor_u2 = {
           "omega": -1 ,
            "torque": -1 ,
            "i": -1 ,
            "u": -1 ,
     }
     expectedHighfor_u2 = {
           "omega": 1 ,
            "torque": 1 ,
            "i": 1 ,
            "u": 1 ,
     }
     assert expectedLowfor_u1 == concreteDcPermanentlyExcitedMotor.get_state_space(input_current,u1)[0]
     assert expectedHighfor_u1 == concreteDcPermanentlyExcitedMotor.get_state_space(input_current,u1)[1]
     assert expectedLowfor_u2 == concreteDcPermanentlyExcitedMotor.get_state_space(input_current,u2)[0]
     assert expectedHighfor_u2 == concreteDcPermanentlyExcitedMotor.get_state_space(input_current,u2)[1]

def test_DcPermanentlyExcitedMotor_reset(concreteDcPermanentlyExcitedMotor):
    new_initial_state = {"i": 100.0}
    default_initial_state = {"i": 10.0}
    default_Initial_state_array = [10.0]
    ex_state_positions = {
          "omega": 0,
          "torque": 1,
          "i":2,
           "u":3
    }
     
    ex_state_space = Box(low=-1, high=1, shape=(4,), dtype=np.float64) 
    assert concreteDcPermanentlyExcitedMotor._initial_states == default_initial_state
    concreteDcPermanentlyExcitedMotor._initial_states = new_initial_state
    assert concreteDcPermanentlyExcitedMotor._initial_states == new_initial_state
    assert np.array_equal(concreteDcPermanentlyExcitedMotor.reset(ex_state_space,ex_state_positions),default_Initial_state_array)
    assert concreteDcPermanentlyExcitedMotor._initial_states == default_initial_state

    """
    default motor reset 

    """
    defaultDcPermanentlyExcitedMotor = DcPermanentlyExcitedMotor()
    default_initialState = {"i": 0.0}
    default_InitialState_array = [0.0]
    
    assert defaultDcPermanentlyExcitedMotor._initial_states == default_initialState
    defaultDcPermanentlyExcitedMotor._initial_states = new_initial_state
    assert defaultDcPermanentlyExcitedMotor._initial_states == new_initial_state
    assert np.array_equal(defaultDcPermanentlyExcitedMotor.reset(ex_state_space,ex_state_positions),default_InitialState_array)
    assert defaultDcPermanentlyExcitedMotor._initial_states == default_initialState

def test_InitDCSeriesMotor(concreteDCSeriesMotor):
    defaultDCSeriesMotor = DcSeriesMotor()
    assert defaultDCSeriesMotor.motor_parameter == {"r_a": 16e-3, "r_e": 48e-3, "l_a": 19e-6, "l_e_prime": 1.7e-3, "l_e": 5.4e-3, "j_rotor": 0.0025,}
    assert defaultDCSeriesMotor._default_initializer == {"states": {"i": 0.0}, "interval": None, "random_init": None, "random_params": (None, None),}
    assert defaultDCSeriesMotor.nominal_values == dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
    assert defaultDCSeriesMotor.limits == dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)
    assert defaultDCSeriesMotor.HAS_JACOBIAN
    assert defaultDCSeriesMotor.I_IDX == 0
    assert defaultDCSeriesMotor.CURRENTS_IDX[0] == 0
    assert concreteDCSeriesMotor.motor_parameter == test_DcSeriesMotor_parameter
    assert concreteDCSeriesMotor._default_initializer == defaultDCSeriesMotor._default_initializer
    assert concreteDCSeriesMotor.initializer == test_DcSeriesMotor_initializer
    assert concreteDCSeriesMotor.nominal_values == defaultDCSeriesMotor.nominal_values
    assert concreteDCSeriesMotor.limits == defaultDCSeriesMotor.limits
    assert concreteDCSeriesMotor._initial_states == test_DcSeriesMotor_initializer["states"]

def test_DCSeriesMotor_Torque(concreteDCSeriesMotor):
     
     currents = [100, 10, 2, 3]
     expectedtorque = concreteDCSeriesMotor.motor_parameter["l_e_prime"]* currents[0] * currents[0]
     assert concreteDCSeriesMotor.torque(currents) == expectedtorque

def test_DCSeriesMotor_el_ode(concreteDCSeriesMotor):
     state = [10.0,15.0]
     u_in = [10,20]
     omega = 60
     expectedODE = np.matmul(concreteDCSeriesMotor._model_constants,np.array([state[0], omega * state[0], u_in[0]]),)
     assert np.array_equal(concreteDCSeriesMotor.electrical_ode(state,u_in,omega),expectedODE)

"""
does i_in function will work because of the index [0] instead 0
def test_DCSeriesMotor_i_in(concreteDCSeriesMotor): 
     currents = [100, 10, 2, 3]
     expectedcurrent = currents[concreteDCSeriesMotor.CURRENTS_IDX]
     assert concreteDCSeriesMotor.i_in(currents) == expectedcurrent
"""

def test_DCSeriesMotor_get_state_space(concreteDCSeriesMotor):
     expectedHigh = {"omega": 1, "torque": 1, "i": 1, "u": 1,}
     u1 = Box(0, 1, shape=(2,), dtype=np.float64)
     i1 = Box(0, 1, shape=(2,), dtype=np.float64)
     u2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     i2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     expectedLowfor_u1_i1 = {"omega": 0, "torque": 0, "i": 0, "u": 0,}
     expectedLowfor_u1_i2 = {"omega": 0, "torque": 0, "i": -1, "u": 0,}
     expectedLowfor_u2_i1 = {"omega": 0, "torque": 0, "i": 0, "u": -1,}
     expectedLowfor_u2_i2 = {"omega": 0, "torque": 0, "i": -1, "u": -1,}

     assert concreteDCSeriesMotor.get_state_space(i1,u1)[0] == expectedLowfor_u1_i1
     assert concreteDCSeriesMotor.get_state_space(i1,u1)[1] == expectedHigh
     assert concreteDCSeriesMotor.get_state_space(i2,u1)[0] == expectedLowfor_u1_i2
     assert concreteDCSeriesMotor.get_state_space(i2,u1)[1] == expectedHigh
     assert concreteDCSeriesMotor.get_state_space(i1,u2)[0] == expectedLowfor_u2_i1
     assert concreteDCSeriesMotor.get_state_space(i1,u2)[1] == expectedHigh
     assert concreteDCSeriesMotor.get_state_space(i2,u2)[0] == expectedLowfor_u2_i2
     assert concreteDCSeriesMotor.get_state_space(i2,u2)[1] == expectedHigh

def test_DCSeriesMotor_el_jacobian(concreteDCSeriesMotor):
     defaultDCSeriesMotor = DcSeriesMotor()
     mp = concreteDCSeriesMotor._motor_parameter
     default_mp = concreteDCSeriesMotor._default_motor_parameter
     state = [10.0,15.0]
     u_in = [10,20]
     omega = 60
     
     expectedJacobian = (
            np.array([[-(mp["r_a"] + mp["r_e"] + mp["l_e_prime"] * omega) / (mp["l_a"] + mp["l_e"])]]),
            np.array([-mp["l_e_prime"] * state[concreteDCSeriesMotor.I_IDX] / (mp["l_a"] + mp["l_e"])]),
            np.array([2 * mp["l_e_prime"] * state[concreteDCSeriesMotor.I_IDX]]),
        )
     default_jacobian = (
            np.array([[-(default_mp["r_a"] + default_mp["r_e"] + default_mp["l_e_prime"] * omega) / (default_mp["l_a"] + default_mp["l_e"])]]),
            np.array([-default_mp["l_e_prime"] * state[concreteDCSeriesMotor.I_IDX] / (default_mp["l_a"] + default_mp["l_e"])]),
            np.array([2 * default_mp["l_e_prime"] * state[concreteDCSeriesMotor.I_IDX]]),
        )
     assert expectedJacobian[0] == concreteDCSeriesMotor.electrical_jacobian(state,u_in,omega)[0]
     assert expectedJacobian[1] == concreteDCSeriesMotor.electrical_jacobian(state,u_in,omega)[1]
     assert expectedJacobian[2] == concreteDCSeriesMotor.electrical_jacobian(state,u_in,omega)[2]
     assert default_jacobian[0] == defaultDCSeriesMotor.electrical_jacobian(state,u_in,omega)[0]
     assert default_jacobian[1] == defaultDCSeriesMotor.electrical_jacobian(state,u_in,omega)[1]
     assert default_jacobian[2] == defaultDCSeriesMotor.electrical_jacobian(state,u_in,omega)[2]

def test_DCSeriesMotor_reset(concreteDCSeriesMotor):
     default_initial_state = {"i": 5.0}
     default_Initial_state_array = [5.0]
     new_initial_state = {"i": 100.0}
     series_state_positions = {"omega": 0,"torque": 1,"i":2, "i_a":3, "i_e":4, "u":5, "u_a":6, "u_e":7}
     series_state_space = Box(low=-1, high=1, shape=(8,), dtype=np.float64) 
     assert concreteDCSeriesMotor._initial_states == default_initial_state
     concreteDCSeriesMotor._initial_states = new_initial_state
     assert concreteDCSeriesMotor._initial_states == new_initial_state
     assert np.array_equal(concreteDCSeriesMotor.reset(series_state_space,series_state_positions),default_Initial_state_array)
     
def test_InitDCShuntMotor(concreteDCShuntMotor):
     defaultDCShuntMotor = DcShuntMotor()

     assert defaultDCShuntMotor._default_motor_parameter =={"r_a": 16e-3, "r_e": 4e-1, "l_a": 19e-6, "l_e_prime": 1.7e-3, "l_e": 5.4e-3, "j_rotor": 0.0025,}
     assert defaultDCShuntMotor._default_initializer == {"states": {"i_a": 0.0, "i_e": 0.0},"interval": None,"random_init": None,"random_params": (None, None),}
     assert defaultDCShuntMotor.HAS_JACOBIAN
     assert defaultDCShuntMotor._default_motor_parameter == defaultDCShuntMotor.motor_parameter
     assert defaultDCShuntMotor._default_initializer  == defaultDCShuntMotor.initializer
     assert defaultDCShuntMotor._default_nominal_values == dict(omega=300, torque=16.0, i=97, i_a=97, i_e=97, u=60, u_a=60, u_e=60)
     assert defaultDCShuntMotor._default_limits == dict(omega=400, torque=38.0, i=210, i_a=210, i_e=210, u=60, u_a=60, u_e=60)
     assert concreteDCShuntMotor.motor_parameter == test_DcShuntMotor_parameter
     assert concreteDCShuntMotor._default_motor_parameter ==  defaultDCShuntMotor._default_motor_parameter
     assert concreteDCShuntMotor.initializer == test_DcShuntMotor_initializer
     assert concreteDCShuntMotor._default_initializer == defaultDCShuntMotor._default_initializer
     assert concreteDCShuntMotor._initial_states == test_DcShuntMotor_initializer["states"]
     assert concreteDCShuntMotor._default_nominal_values == defaultDCShuntMotor._default_nominal_values
     assert concreteDCShuntMotor._default_limits == defaultDCShuntMotor._default_limits
     assert concreteDCShuntMotor.I_A_IDX == 0
     assert concreteDCShuntMotor.I_E_IDX == 1

def test_DCShuntMotor_i_in(concreteDCShuntMotor):
     defaultDCShuntMotor = DcShuntMotor()
     currents = [5, 10, 20, 25]
     expectedexpectedcurrent = [5 + 10]
     assert concreteDCShuntMotor.i_in(currents) == expectedexpectedcurrent
     assert defaultDCShuntMotor.i_in(currents) == concreteDCShuntMotor.i_in(currents)

def test_DCShuntMotor_el_ode(concreteDCShuntMotor):
     state = [5, 10, 20, 25]
     u_in = [10,20]
     omega = 60
     expectedOde = np.matmul(concreteDCShuntMotor._model_constants,np.array([state[0],state[1],omega * state[1],u_in[0],u_in[0],]),)
     assert np.array_equal(concreteDCShuntMotor.electrical_ode(state,u_in,omega),expectedOde)

def test_DCShuntMotor_el_jacobian(concreteDCShuntMotor):
     defaultDCShuntMotor = DcShuntMotor()
     state = [5, 10, 20, 25]
     u_in = [10,20]
     omega = 60
     mp = concreteDCShuntMotor._motor_parameter
     default_mp = concreteDCShuntMotor._default_motor_parameter
     expectedJacobian = (
          np.array([[-mp["r_a"] / mp["l_a"], -mp["l_e_prime"] / mp["l_a"] * omega],[0, -mp["r_e"] / mp["l_e"]],]),
          np.array([-mp["l_e_prime"] * state[1] / mp["l_a"], 0]),
          np.array([mp["l_e_prime"] * state[1],mp["l_e_prime"] * state[0],]),
          )
     defaultJacobian = (
          np.array([[-default_mp["r_a"] / default_mp["l_a"], -default_mp["l_e_prime"] / default_mp["l_a"] * omega],[0, -default_mp["r_e"] / default_mp["l_e"]],]),
          np.array([-default_mp["l_e_prime"] * state[1] / default_mp["l_a"], 0]),
          np.array([default_mp["l_e_prime"] * state[1],default_mp["l_e_prime"] * state[0],]),
          )
     assert np.array_equal(expectedJacobian[0], concreteDCShuntMotor.electrical_jacobian(state,u_in,omega)[0])
     assert np.array_equal(expectedJacobian[1], concreteDCShuntMotor.electrical_jacobian(state,u_in,omega)[1])
     assert np.array_equal(expectedJacobian[2], concreteDCShuntMotor.electrical_jacobian(state,u_in,omega)[2])
     assert np.array_equal(defaultJacobian[0], defaultDCShuntMotor.electrical_jacobian(state,u_in,omega)[0])
     assert np.array_equal(defaultJacobian[1], defaultDCShuntMotor.electrical_jacobian(state,u_in,omega)[1])
     assert np.array_equal(defaultJacobian[2], defaultDCShuntMotor.electrical_jacobian(state,u_in,omega)[2])

def test_DCShuntMotor_get_state_space(concreteDCShuntMotor):
     expectedHigh = {"omega": 1, "torque": 1,  "i_a": 1,"i_e": 1,"u": 1,}
     u1 = Box(0, 1, shape=(2,), dtype=np.float64)
     i1 = Box(0, 1, shape=(2,), dtype=np.float64)
     u2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     i2 = Box(-1, 1, shape=(2,), dtype=np.float64)
     expectedLowfor_u1_i1 = {"omega": 0, "torque": 0,  "i_a": 0,"i_e": 0,"u": 0,}
     expectedLowfor_u1_i2 = {"omega": 0, "torque": -1,  "i_a": -1,"i_e": -1,"u": 0,}
     expectedLowfor_u2_i1 = {"omega": 0, "torque": 0,  "i_a": 0,"i_e": 0,"u": -1,}
     expectedLowfor_u2_i2 = {"omega": 0, "torque": -1,  "i_a": -1,"i_e": -1,"u": -1,} 

     assert concreteDCShuntMotor.get_state_space(i1,u1)[0] == expectedLowfor_u1_i1
     assert concreteDCShuntMotor.get_state_space(i1,u1)[1] == expectedHigh
     assert concreteDCShuntMotor.get_state_space(i2,u1)[0] == expectedLowfor_u1_i2
     assert concreteDCShuntMotor.get_state_space(i2,u1)[1] == expectedHigh
     assert concreteDCShuntMotor.get_state_space(i1,u2)[0] == expectedLowfor_u2_i1
     assert concreteDCShuntMotor.get_state_space(i1,u2)[1] == expectedHigh
     assert concreteDCShuntMotor.get_state_space(i2,u2)[0] == expectedLowfor_u2_i2
     assert concreteDCShuntMotor.get_state_space(i2,u2)[1] == expectedHigh