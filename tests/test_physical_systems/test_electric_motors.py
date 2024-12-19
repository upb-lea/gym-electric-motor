import math
import pytest
import gym_electric_motor as gem
from gym_electric_motor.physical_systems.electric_motors import (
    ElectricMotor,
    DcMotor,
    DcExternallyExcitedMotor,
    DcPermanentlyExcitedMotor,
    DcSeriesMotor,
    DcShuntMotor,
    ThreePhaseMotor,
    InductionMotor,
    DoublyFedInductionMotor,
    SquirrelCageInductionMotor,
    SynchronousMotor,
    SynchronousReluctanceMotor,
    ExternallyExcitedSynchronousMotor,
    PermanentMagnetSynchronousMotor
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
test_Induction_motor_parameter = {"p": 3,"l_m": 143.75e-3,"l_sigs": 5.87e-3,"l_sigr": 5.87e-2,"j_rotor": 1.1e-3,"r_s": 3,"r_r": 1.355,}
test_DcInductionMotor_initializer = {"states": {"i_salpha": 1.0,"i_sbeta": 0.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 0.0,},"interval": None,"random_init": None,"random_params": (None, None),}
test_ReluctanceMotor_parameter = {"p": 4,"l_d": 1e-3,"l_q": 1e-3,"j_rotor": 1e-3,"r_s": 0.5,}
test_ReluctanceMotor_initializer = {
        "states": {"i_sq": 1.0, "i_sd": 2.0, "epsilon": 10.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
test_ExtExcSyncMotor_parameter = {'p': 3,'l_d': 1.66e-1,'l_q': 0.35e-1,'l_m': 1.589e-5,'l_e': 1.74e-3,'j_rotor': 0.3883,'r_s': 15.55e-3,'r_e': 7.2e-3,'k': 65,}
test_ExtExcSyncMotor_initializer = {
        "states": {"i_sq": 2.0, "i_sd": 1.0, "i_e": 2.0, "epsilon": 2.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
test_PermMagSyncMotor_parameter = {"p": 3,"l_d": 0.37e-6,"l_q": 1.2e-6,"j_rotor": 0.03883,"r_s": 18,"psi_p": 66e-6,}
test_PermMagSyncMotor_initializer = {"states": {"i_sq": 10.0, "i_sd": 5.0, "epsilon": 10.0},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
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

@pytest.fixture
def concreteInductionMotor():
    """
     pytest fixture that returns a Dc Shunt motor object
    :return:  Dc Shunt motor object with concrete values
    """
    return InductionMotor(test_Induction_motor_parameter,None,None,test_DcInductionMotor_initializer )

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
    concreteDCMotor.initializer["states"] = new_initial_state
    concreteDCMotor.reset(extex_state_space,extex_state_positions)
    assert concreteDCMotor._initial_states == new_initial_state

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
     defaultDCMotor.initializer["states"] = new_initial_state
     defaultDCMotor.reset(extex_state_space,extex_state_positions)
     assert defaultDCMotor._initial_states == new_initial_state

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

def test_DcPermanentlyExcitedMotor_i_in(concreteDcPermanentlyExcitedMotor):
     state = np.array([100,200,300])
     expected_return = state[concreteDcPermanentlyExcitedMotor.CURRENTS_IDX]
     assert concreteDcPermanentlyExcitedMotor.i_in(state) == expected_return


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
    concreteDcPermanentlyExcitedMotor.initializer["states"] = new_initial_state
    concreteDcPermanentlyExcitedMotor.reset(ex_state_space,ex_state_positions)
    assert concreteDcPermanentlyExcitedMotor._initial_states == new_initial_state
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
    defaultDcPermanentlyExcitedMotor.initializer["states"] = new_initial_state
    defaultDcPermanentlyExcitedMotor.reset(ex_state_space,ex_state_positions)
    assert defaultDcPermanentlyExcitedMotor._initial_states == new_initial_state

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

def test_DCSeriesMotor_i_in(concreteDCSeriesMotor): 
     currents = np.array([100, 10, 2, 3])
     expectedcurrent = currents[concreteDCSeriesMotor.CURRENTS_IDX]
     assert concreteDCSeriesMotor.i_in(currents) == expectedcurrent


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
     concreteDCSeriesMotor.initializer["states"] = new_initial_state
     concreteDCSeriesMotor.reset(series_state_space,series_state_positions)
     assert concreteDCSeriesMotor._initial_states == new_initial_state
     
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
     expectedcurrent = [5 + 10]
     assert concreteDCShuntMotor.i_in(currents) == expectedcurrent
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

def test_ThreePhaseMotor_t_23():
     defaultThreePhaseMotor = ThreePhaseMotor()
     t23 = 2 / 3 * np.array([
        [1, -0.5, -0.5],
        [0, 0.5 * np.sqrt(3), -0.5 * np.sqrt(3)]
    ])
     u_abc = [1, 2, 3] #[u_a, u_b, u_c]
     expectedU_out = np.matmul(t23, u_abc) # [u_alpha, u_beta]
     assert np.array_equal(defaultThreePhaseMotor.t_23(u_abc),expectedU_out)

def test_ThreePhaseMotor_t_32():
     defaultThreePhaseMotor = ThreePhaseMotor()
     t32 = np.array([
        [1, 0],
        [-0.5, 0.5 * np.sqrt(3)],
        [-0.5, -0.5 * np.sqrt(3)]
    ])
     u_alpha_beta = [10,20] # [u_alpha, u_beta]
     expectedU_out = np.matmul(t32, u_alpha_beta) # [u_a, u_b, u_c]
     assert np.array_equal(defaultThreePhaseMotor.t_32(u_alpha_beta),expectedU_out)

def test_ThreePhaseMotor_q():
     defaultThreePhaseMotor = ThreePhaseMotor()
     epsilon = 15 # Current electrical angle of the motor
     input_dq = [10,20]
     cos = math.cos(epsilon)
     sin = math.sin(epsilon)
     out_alpha_beta = cos * input_dq[0] - sin * input_dq[1], sin * input_dq[0] + cos * input_dq[1]
     assert defaultThreePhaseMotor.q(input_dq,epsilon) == out_alpha_beta

def test_ThreePhaseMotor_q_inv():
     defaultThreePhaseMotor = ThreePhaseMotor()
     epsilon = 15 # Current electrical angle of the motor
     e = -epsilon
     input_alpha_beta = [10,20]
     cos = math.cos(e)
     sin = math.sin(e)
     out_dq =  cos * input_alpha_beta[0] - sin * input_alpha_beta[1], sin * input_alpha_beta[0] + cos * input_alpha_beta[1]
     assert defaultThreePhaseMotor.q_inv(input_alpha_beta,epsilon) == out_dq

def test_ThreePhaseMotor_q_me():
     defaultInductionMotorMotor = InductionMotor(test_Induction_motor_parameter,None,None,None,None)
     epsilon =  10 #Current mechanical angle of the motor
     e = epsilon* defaultInductionMotorMotor._motor_parameter["p"]
     input_dq = [10,20]
     cos = math.cos(e)
     sin = math.sin(e)
     out_alpha_beta = cos * input_dq[0] - sin * input_dq[1], sin * input_dq[0] + cos * input_dq[1]
     assert defaultInductionMotorMotor.q_me(input_dq,epsilon) == out_alpha_beta

def test_ThreePhaseMotor_q_inv_me():
     defaultInductionMotorMotor = InductionMotor(test_Induction_motor_parameter,None,None,None,None)
     epsilon =  10 #Current mechanical angle of the motor
     e = -epsilon* defaultInductionMotorMotor._motor_parameter["p"]
     input_dq = [10,20]
     cos = math.cos(e)
     sin = math.sin(e)
     out_alpha_beta = cos * input_dq[0] - sin * input_dq[1], sin * input_dq[0] + cos * input_dq[1]
     assert defaultInductionMotorMotor.q_inv_me(input_dq,epsilon) == out_alpha_beta

def test_InitInductionMotor(concreteInductionMotor):
     defaultInductionMotor = InductionMotor()

     assert defaultInductionMotor._default_motor_parameter =={"p": 2,"l_m": 143.75e-3,"l_sigs": 5.87e-3,"l_sigr": 5.87e-3,"j_rotor": 1.1e-3,"r_s": 2.9338,"r_r": 1.355,}
     assert defaultInductionMotor._default_initializer == {"states": {"i_salpha": 0.0,"i_sbeta": 0.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 0.0,},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
     assert defaultInductionMotor.HAS_JACOBIAN 
     assert defaultInductionMotor._default_nominal_values == dict(omega=3e3 * np.pi / 30, torque=0.0, i=3.9, epsilon=math.pi, u=560)
     assert defaultInductionMotor._default_limits == dict(omega=4e3 * np.pi / 30, torque=0.0, i=5.5, epsilon=math.pi, u=560)
     assert concreteInductionMotor.motor_parameter == test_Induction_motor_parameter
     assert concreteInductionMotor._default_motor_parameter ==  defaultInductionMotor._default_motor_parameter
     assert concreteInductionMotor._default_motor_parameter ==  defaultInductionMotor.motor_parameter
     assert concreteInductionMotor._initial_states == test_DcInductionMotor_initializer["states"]
     assert concreteInductionMotor._default_nominal_values == defaultInductionMotor._default_nominal_values
     assert concreteInductionMotor._default_limits == defaultInductionMotor._default_limits
     assert defaultInductionMotor.I_SALPHA_IDX == 0
     assert defaultInductionMotor.I_SBETA_IDX  == 1

def test_InductionMotor_el_ode(concreteInductionMotor):
     state = [5, 10, 15, 20, 10] #[i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
     u_in = np.array([(10,11), (1,2)]) #[u_salpha, u_sbeta, u_ralpha, u_rbeta]
     omega = 60
     expectedOde = np.matmul(
            concreteInductionMotor._model_constants,
            np.array(
                [
                    # omega, i_alpha, i_beta, psi_ralpha, psi_rbeta, omega * psi_ralpha, omega * psi_rbeta, u_salpha, u_sbeta, u_ralpha, u_rbeta,
                    omega,
                    state[concreteInductionMotor.I_SALPHA_IDX],
                    state[concreteInductionMotor.I_SBETA_IDX],
                    state[concreteInductionMotor.PSI_RALPHA_IDX],
                    state[concreteInductionMotor.PSI_RBETA_IDX],
                    omega * state[concreteInductionMotor.PSI_RALPHA_IDX],
                    omega * state[concreteInductionMotor.PSI_RBETA_IDX],
                    u_in[0, 0],
                    u_in[0, 1],
                    u_in[1, 0],
                    u_in[1, 1],
                ]
            ),
        )
     assert np.array_equal(expectedOde, concreteInductionMotor.electrical_ode(state,u_in,omega))

def test_InductionMotor_i_in(concreteInductionMotor):
     state = np.array([(1, 2), (10, 11), (20,21)])
     expectedcurrent = np.array([(1, 2), (10, 11)])
     assert np.array_equal(expectedcurrent,concreteInductionMotor.i_in(state))

def test_InductionMotor_torque(concreteInductionMotor):
      mp = concreteInductionMotor._motor_parameter
      states = [5, 10, 15, 20, 10] #[i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
      expectedTorque = (
            1.5
            * mp["p"]
            * mp["l_m"]
            / (mp["l_m"] + mp["l_sigr"])
            * (
                states[concreteInductionMotor.PSI_RALPHA_IDX] * states[concreteInductionMotor.I_SBETA_IDX]
                - states[concreteInductionMotor.PSI_RBETA_IDX] * states[concreteInductionMotor.I_SALPHA_IDX]
            )
        )
      assert np.array_equal(concreteInductionMotor.torque(states),expectedTorque)

def test_InductionMotor_el_jacobian(concreteInductionMotor):
        mp = concreteInductionMotor._motor_parameter
        l_s = mp["l_m"] + mp["l_sigs"]
        l_r = mp["l_m"] + mp["l_sigr"]
        sigma = (l_s * l_r - mp["l_m"] ** 2) / (l_s * l_r)
        tau_r = l_r / mp["r_r"]
        tau_sig = sigma * l_s / (mp["r_s"] + mp["r_r"] * (mp["l_m"] ** 2) / (l_r**2))
        state = [5, 10, 15, 20, 10] #[i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
        u_in = np.array([(10,11), (1,2)])#[u_salpha, u_sbeta, u_ralpha, u_rbeta]
        omega = 60

        expectedJacobian = (
            np.array(
                [  # dx'/dx
                    # i_alpha          i_beta               psi_alpha                                    psi_beta                                   epsilon
                    [
                        -1 / tau_sig,
                        0,
                        mp["l_m"] * mp["r_r"] / (sigma * l_s * l_r**2),
                        omega * mp["l_m"] * mp["p"] / (sigma * l_r * l_s),
                        0,
                    ],
                    [
                        0,
                        -1 / tau_sig,
                        -omega * mp["l_m"] * mp["p"] / (sigma * l_r * l_s),
                        mp["l_m"] * mp["r_r"] / (sigma * l_s * l_r**2),
                        0,
                    ],
                    [mp["l_m"] / tau_r, 0, -1 / tau_r, -omega * mp["p"], 0],
                    [0, mp["l_m"] / tau_r, omega * mp["p"], -1 / tau_r, 0],
                    [0, 0, 0, 0, 0],
                ]
            ),
            np.array(
                [  # dx'/dw
                    mp["l_m"] * mp["p"] / (sigma * l_r * l_s) * state[concreteInductionMotor.PSI_RBETA_IDX],
                    -mp["l_m"] * mp["p"] / (sigma * l_r * l_s) * state[concreteInductionMotor.PSI_RALPHA_IDX],
                    -mp["p"] * state[concreteInductionMotor.PSI_RBETA_IDX],
                    mp["p"] * state[concreteInductionMotor.PSI_RALPHA_IDX],
                    mp["p"],
                ]
            ),
            np.array(
                [  # dT/dx
                    -state[concreteInductionMotor.PSI_RBETA_IDX] * 3 / 2 * mp["p"] * mp["l_m"] / l_r,
                    state[concreteInductionMotor.PSI_RALPHA_IDX] * 3 / 2 * mp["p"] * mp["l_m"] / l_r,
                    state[concreteInductionMotor.I_SBETA_IDX] * 3 / 2 * mp["p"] * mp["l_m"] / l_r,
                    -state[concreteInductionMotor.I_SALPHA_IDX] * 3 / 2 * mp["p"] * mp["l_m"] / l_r,
                    0,
                ]
            ),
        )

        assert np.array_equal(expectedJacobian[0],concreteInductionMotor.electrical_jacobian(state,u_in,omega)[0])
        assert np.array_equal(expectedJacobian[1],concreteInductionMotor.electrical_jacobian(state,u_in,omega)[1])
        assert np.array_equal(expectedJacobian[2],concreteInductionMotor.electrical_jacobian(state,u_in,omega)[2])


"""
def test_InductionMotor_reset(concreteInductionMotor):
     #_nominal_values ---> _initial_limits
     new_initial_state = {"i_salpha": 5.0,"i_sbeta": 6.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 10.0,}
     default_initial_state = {"i_salpha": 1.0,"i_sbeta": 0.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 0.0,}
     default_initial_state_array = [1,0,0,0,0]
     InductionMotor_state_positions = {
          "i_salpha": 0,
          "i_sbeta": 1,
          "psi_ralpha": 2,
          "psi_rbeta": 3,
          "omega": 4,
          "torque": 5,
          "epsilon":6,
          "u": 7
    }
     assert concreteInductionMotor._initial_states == default_initial_state
     concreteInductionMotor._initial_states = new_initial_state 
     assert concreteInductionMotor._initial_states == new_initial_state
     InductionMotor_state_space = Box(low=-1, high=1, shape=(8,), dtype=np.float64)
     assert np.array_equal(concreteInductionMotor.reset(InductionMotor_state_space,InductionMotor_state_positions),default_initial_state_array)
"""
def test_InitDoublyFedIM():
     defaultDoublyFedIM = DoublyFedInductionMotor()

     assert defaultDoublyFedIM.motor_parameter == {"p": 2,"l_m": 297.5e-3,"l_sigs": 25.71e-3,"l_sigr": 25.71e-3,"j_rotor": 13.695e-3,"r_s": 4.42,"r_r": 3.51,}
     assert defaultDoublyFedIM._default_initializer == {"states": {"i_salpha": 0.0,"i_sbeta": 0.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 0.0,},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
     assert defaultDoublyFedIM._default_nominal_values == dict(omega=1650 * np.pi / 30, torque=0.0, i=7.5, epsilon=math.pi, u=720)
     assert defaultDoublyFedIM._default_limits == dict(omega=1800 * np.pi / 30, torque=0.0, i=9, epsilon=math.pi, u=720)
     assert defaultDoublyFedIM.IO_CURRENTS == (["i_sa", "i_sb", "i_sc", "i_salpha", "i_sbeta", "i_sd", "i_sq"] + defaultDoublyFedIM.IO_ROTOR_CURRENTS)
     assert defaultDoublyFedIM.HAS_JACOBIAN

def test_InitSquirrelCageIM():
     defaultSquirrelCageIM = SquirrelCageInductionMotor()
     defaultDoublyFedIM = DoublyFedInductionMotor()
     assert defaultSquirrelCageIM.motor_parameter == {"p": 2,"l_m": 143.75e-3,"l_sigs": 5.87e-3,"l_sigr": 5.87e-3,"j_rotor": 1.1e-3,"r_s": 2.9338,"r_r": 1.355,}
     assert defaultSquirrelCageIM._default_initializer == defaultDoublyFedIM._default_initializer
     assert defaultSquirrelCageIM._default_limits ==  dict(omega=4e3 * np.pi / 30, torque=0.0, i=5.5, epsilon=math.pi, u=560)
     new_motor_parameter = {"p": 3,"l_m": 140,"l_sigs": 5.87e-3,"l_sigr": 5.87e-2,"j_rotor": 1.1e-3,"r_s": 2.9338,"r_r": 1.1,}
     initializer = {"states": {"i_salpha": 0.0,"i_sbeta": 0.0,"psi_ralpha": 0.0,"psi_rbeta": 0.0,"epsilon": 0.0,},
        "interval": None,
        "random_init": None,
        "random_params": (None, None),
    }
     concreteSquirrelCageIM = SquirrelCageInductionMotor(new_motor_parameter,None,None,initializer,None)
     assert concreteSquirrelCageIM.initializer == initializer
     assert concreteSquirrelCageIM.motor_parameter == new_motor_parameter

def test_SquirrelCageIM_el_ode():
     defaultSquirrelCageIM = SquirrelCageInductionMotor()
     state = [5, 10, 15, 20, 10] #[i_salpha, i_sbeta, psi_ralpha, psi_rbeta, epsilon]
     u_salphabeta = [10,11]
     u_in = np.array([(10,11), (0,0)])
     omega = 60
     expectedOde = np.matmul(
            defaultSquirrelCageIM._model_constants,
            np.array(
                [
                    omega,
                    state[defaultSquirrelCageIM.I_SALPHA_IDX],
                    state[defaultSquirrelCageIM.I_SBETA_IDX],
                    state[defaultSquirrelCageIM.PSI_RALPHA_IDX],
                    state[defaultSquirrelCageIM.PSI_RBETA_IDX],
                    omega * state[defaultSquirrelCageIM.PSI_RALPHA_IDX],
                    omega * state[defaultSquirrelCageIM.PSI_RBETA_IDX],
                    u_in[0, 0],
                    u_in[0, 1],
                    u_in[1, 0],
                    u_in[1, 1],
                ]
            ),
        )
     assert np.array_equal(expectedOde, defaultSquirrelCageIM.electrical_ode(state,u_salphabeta,omega))

# ._update_model() is not implemented and hence cant be initialized - discuss
def test_InitSynchronousMotor():
     with pytest.raises(NotImplementedError):
          SynchronousMotor()

def test_InitSynchronousReluctanceMotor():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     assert defaultReluctanceMotor._default_motor_parameter == {"p": 4,"l_d": 10.1e-3,"l_q": 4.1e-3,"j_rotor": 0.8e-3,"r_s": 0.57,}
     assert defaultReluctanceMotor.HAS_JACOBIAN
     assert defaultReluctanceMotor.motor_parameter == defaultReluctanceMotor._default_motor_parameter
     assert defaultReluctanceMotor._initial_states == defaultReluctanceMotor._default_initializer["states"]
     concreteReluctanceMotor = SynchronousReluctanceMotor(test_ReluctanceMotor_parameter,None,None,test_ReluctanceMotor_initializer)
     assert concreteReluctanceMotor.motor_parameter == test_ReluctanceMotor_parameter
     assert concreteReluctanceMotor._initializer == test_ReluctanceMotor_initializer
     assert concreteReluctanceMotor._default_motor_parameter == defaultReluctanceMotor._default_motor_parameter
     assert concreteReluctanceMotor._initial_states == test_ReluctanceMotor_initializer["states"]
     assert defaultReluctanceMotor.I_SD_IDX == 0
     assert defaultReluctanceMotor.I_SQ_IDX == 1
     assert defaultReluctanceMotor.EPSILON_IDX == 2
     assert defaultReluctanceMotor.CURRENTS == ["i_sd", "i_sq"]
     assert defaultReluctanceMotor.VOLTAGES == ["u_sd", "u_sq"]

def test_SyncReluctanceMotor_torque():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     mp = defaultReluctanceMotor._motor_parameter
     currents = [10,15]
     expectedTorque = 1.5 * mp["p"] * ((mp["l_d"] - mp["l_q"]) * currents[defaultReluctanceMotor.I_SD_IDX]) * currents[defaultReluctanceMotor.I_SQ_IDX]
     assert np.array_equal(defaultReluctanceMotor.torque(currents),expectedTorque)

def test_SyncReluctanceMotor_el_jacobian():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     state = [2,4,10]#[i_sd, i_sq, epsilon]
     omega = 60
     u_in = [5,10]#[u_sd, u_sq]
     mp = defaultReluctanceMotor._motor_parameter
     expectedJacobian = (
            np.array(
                [
                    [
                        -mp["r_s"] / mp["l_d"],
                        mp["l_q"] / mp["l_d"] * mp["p"] * omega,
                        0,
                    ],
                    [
                        -mp["l_d"] / mp["l_q"] * mp["p"] * omega,
                        -mp["r_s"] / mp["l_q"],
                        0,
                    ],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [
                    mp["p"] * mp["l_q"] / mp["l_d"] * state[defaultReluctanceMotor.I_SQ_IDX],
                    -mp["p"] * mp["l_d"] / mp["l_q"] * state[defaultReluctanceMotor.I_SD_IDX],
                    mp["p"],
                ]
            ),
            np.array(
                [
                    1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[defaultReluctanceMotor.I_SQ_IDX],
                    1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[defaultReluctanceMotor.I_SD_IDX],
                    0,
                ]
            ),
        )
     assert np.array_equal(expectedJacobian[0],defaultReluctanceMotor.electrical_jacobian(state,u_in,omega)[0])
     assert np.array_equal(expectedJacobian[1],defaultReluctanceMotor.electrical_jacobian(state,u_in,omega)[1])
     assert np.array_equal(expectedJacobian[2],defaultReluctanceMotor.electrical_jacobian(state,u_in,omega)[2])
     
def test_SyncReluctanceMotor_el_ode():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     state = [2,4,10]#[i_sd, i_sq, epsilon]
     omega = 60
     u_in = [5,10]#[u_sd, u_sq]
     expectedOde = np.matmul(
            defaultReluctanceMotor._model_constants,
            np.array(
                [
                    omega,
                    state[defaultReluctanceMotor.I_SD_IDX],
                    state[defaultReluctanceMotor.I_SQ_IDX],
                    u_in[0],
                    u_in[1],
                    omega * state[defaultReluctanceMotor.I_SD_IDX],
                    omega * state[defaultReluctanceMotor.I_SQ_IDX],
                ]
            ),
        )
     assert np.array_equal(defaultReluctanceMotor.electrical_ode(state,u_in,omega),expectedOde)

def test_SyncReluctanceMotor_i_in():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     state = np.array([1,2,3,5,8,9])
     expectedCurrent = [1,2]
     assert np.array_equal(defaultReluctanceMotor.i_in(state),expectedCurrent)

def test_SyncReluctanceMotor_reset():
     defaultReluctanceMotor = SynchronousReluctanceMotor()
     default_initial_state = {"i_sq": 0.0, "i_sd": 0.0, "epsilon": 0.0}
     default_Initial_state_array = [0,0,0]
     new_initial_state = {"i_sq": 1.0, "i_sd": 2.0, "epsilon": 10.0}
     ReluctanceMotor_state_positions = {"i_sq": 0,"i_sd": 1,"torque": 2,"omega": 3,"epsilon": 4,"u": 5,}
     ReluctanceMotor_state_space = Box(low=-1, high=1, shape=(6,), dtype=np.float64)
     assert defaultReluctanceMotor._initial_states == default_initial_state
     defaultReluctanceMotor._initial_states = new_initial_state
     assert defaultReluctanceMotor._initial_states == new_initial_state
     assert np.array_equal(defaultReluctanceMotor.reset(ReluctanceMotor_state_space,ReluctanceMotor_state_positions),default_Initial_state_array)
     defaultReluctanceMotor.initializer["states"] = new_initial_state
     defaultReluctanceMotor.reset(ReluctanceMotor_state_space,ReluctanceMotor_state_positions)
     assert defaultReluctanceMotor._initial_states == new_initial_state

def test_InitExtExcitedSynchronousMotor():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     assert ExtExcSyncMotor._default_motor_parameter == {'p': 3,'l_d': 1.66e-3,'l_q': 0.35e-3,'l_m': 1.589e-3,'l_e': 1.74e-3,'j_rotor': 0.3883,'r_s': 15.55e-3,'r_e': 7.2e-3,'k': 65.21,}
     assert ExtExcSyncMotor.HAS_JACOBIAN
     mp = ExtExcSyncMotor._motor_parameter
     ExtExcSyncMotor._default_motor_parameter.update({'r_E':mp['k'] ** 2 * 3/2 * mp['r_e'], 'l_M':mp['k'] * 3/2 * mp['l_m'], 'l_E': mp['k'] ** 2 * 3/2 * mp['l_e'], 'i_k_rs':2 / 3 / mp['k'], 'sigma':1 - mp['l_M'] ** 2 / (mp['l_d'] * mp['l_E'])})
     assert ExtExcSyncMotor.motor_parameter == ExtExcSyncMotor._default_motor_parameter
     assert ExtExcSyncMotor._initial_states == ExtExcSyncMotor._default_initializer["states"]
     concreteExtExcSyncMotor = ExternallyExcitedSynchronousMotor(test_ExtExcSyncMotor_parameter,None,None,test_ExtExcSyncMotor_initializer)
     newmp = concreteExtExcSyncMotor._motor_parameter
     test_ExtExcSyncMotor_parameter.update({'r_E':newmp['k'] ** 2 * 3/2 * newmp['r_e'], 'l_M':newmp['k'] * 3/2 * newmp['l_m'], 'l_E': newmp['k'] ** 2 * 3/2 * newmp['l_e'], 'i_k_rs':2 / 3 / newmp['k'], 'sigma':1 - newmp['l_M'] ** 2 / (newmp['l_d'] * newmp['l_E'])})
     assert concreteExtExcSyncMotor.motor_parameter == test_ExtExcSyncMotor_parameter
     assert concreteExtExcSyncMotor._initializer == test_ExtExcSyncMotor_initializer
     assert concreteExtExcSyncMotor._default_motor_parameter == ExtExcSyncMotor._default_motor_parameter
     assert concreteExtExcSyncMotor._initial_states == test_ExtExcSyncMotor_initializer["states"]
     assert ExtExcSyncMotor.I_SD_IDX == 0
     assert ExtExcSyncMotor.I_SQ_IDX == 1
     assert ExtExcSyncMotor.EPSILON_IDX == 3
     assert ExtExcSyncMotor.CURRENTS == ["i_sd", "i_sq", "i_e"]
     assert ExtExcSyncMotor.VOLTAGES == ["u_sd", "u_sq", "u_e"]

"""
what should the u_in and how come the u_dqe[2],
"""
def test_ExtExcitedSynchronousMotor_el_ode():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     state = [5, 5, 10]#[i_sd, i_sq, epsilon]
     omega = 60
     u_in = [50, 60, 50]#[u_sd, u_sq]
     expectedOde = np.matmul(
            ExtExcSyncMotor._model_constants,
            np.array(
                [
                    omega,
                    state[ExtExcSyncMotor.I_SD_IDX],
                    state[ExtExcSyncMotor.I_SQ_IDX],
                    state[ExtExcSyncMotor.I_E_IDX],
                    u_in[0],
                    u_in[1],
                    u_in[2],
                    omega * state[ExtExcSyncMotor.I_SD_IDX],
                    omega * state[ExtExcSyncMotor.I_SQ_IDX],
                    omega * state[ExtExcSyncMotor.I_E_IDX]
                ]
            )
        )
     assert np.array_equal(ExtExcSyncMotor.electrical_ode(state,u_in,omega),expectedOde)

def test_ExtExcitedSynchronousMotor_torque():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     mp = ExtExcSyncMotor._motor_parameter
     currents = [2,3,4]#["i_sd", "i_sq", "i_e"]
     expectedTorque = 1.5 * mp["p"] * (mp["l_M"] * currents[ExtExcSyncMotor.I_E_IDX] * mp["i_k_rs"] + (mp["l_d"] - mp["l_q"]) * currents[ExtExcSyncMotor.I_SD_IDX]) * currents[ExtExcSyncMotor.I_SQ_IDX]
     assert np.array_equal(ExtExcSyncMotor.torque(currents),expectedTorque)

def test_ExtExcitedSynchronousMotor_el_jacobian():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     mp = ExtExcSyncMotor._motor_parameter
     state = [5, 5, 10]#[i_sd, i_sq, epsilon]
     omega = 60
     u_in = [50, 60, 50]#[u_sd, u_sq]
     expectedJacobian = (
            np.array([ # dx'/dx
                [                                      -mp["r_s"] / (mp["l_d"] * mp["sigma"]),                                         mp["l_q"] / (mp["sigma"] * mp["l_d"]) * omega * mp["p"], mp["l_M"] * mp["r_E"] / (mp["sigma"] * mp["l_d"] * mp["l_E"]) * mp["i_k_rs"], 0],
                [                                    -mp["l_d"] / mp["l_q"] * omega * mp["p"],                                                                          -mp["r_s"] / mp["l_q"],                      -omega * mp["p"] * mp["l_M"] / mp["l_q"] * mp["i_k_rs"], 0],
                [mp["l_M"] * mp["r_s"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]), -omega * mp["p"] * mp["l_M"] * mp["l_q"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]),                                       -mp["r_E"] / (mp["sigma"] * mp["l_E"]), 0],
                [                                                                           0,                                                                                               0,                                                                            0, 0],
            ]),
            np.array([ # dx'/dw
                mp["p"] * mp["l_q"] / (mp["l_d"] * mp["sigma"]) * state[ExtExcSyncMotor.I_SQ_IDX],
                -mp["p"] * mp["l_d"] / mp["l_q"] * state[ExtExcSyncMotor.I_SD_IDX] - mp["p"] * mp["l_M"] / mp["l_q"] * state[ExtExcSyncMotor.I_E_IDX] * mp["i_k_rs"],
                -mp["p"] * mp["l_M"] * mp["l_q"] / (mp["sigma"] * mp["l_d"] * mp["l_E"] * mp["i_k_rs"]) * state[ExtExcSyncMotor.I_SQ_IDX],
                mp["p"],
            ]),
            np.array([ # dT/dx
                1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[ExtExcSyncMotor.I_SQ_IDX],
                1.5 * mp["p"] * (mp["l_M"] * state[ExtExcSyncMotor.I_E_IDX] * mp["i_k_rs"] + (mp["l_d"] - mp["l_q"]) * state[ExtExcSyncMotor.I_SD_IDX]),
                1.5 * mp["p"] * mp["l_M"] * mp["i_k_rs"] * state[ExtExcSyncMotor.I_SQ_IDX],
                0,
            ])
        ) 
     assert np.array_equal(expectedJacobian[0],ExtExcSyncMotor.electrical_jacobian(state,u_in,omega)[0])
     assert np.array_equal(expectedJacobian[1],ExtExcSyncMotor.electrical_jacobian(state,u_in,omega)[1])
     assert np.array_equal(expectedJacobian[2],ExtExcSyncMotor.electrical_jacobian(state,u_in,omega)[2])

def test_ExtExcitedSynchronousMotor_i_in():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     state = np.array([10,20,3,5,8,9])
     expectedCurrent = [10,20,3]
     assert np.array_equal(ExtExcSyncMotor.i_in(state),expectedCurrent)

def test_ExtExcSyncMotor_reset():
     ExtExcSyncMotor = ExternallyExcitedSynchronousMotor()
     default_initial_state = {"i_sq": 0.0, "i_sd": 0.0, "i_e": 0.0, "epsilon": 0.0}
     default_Initial_state_array = [0,0,0,0]
     new_initial_state = {"i_sq": 10.0, "i_sd": 0.0, "i_e": 10.0, "epsilon": 5.0}
     ExtExcSyncMotor_state_positions = {"i_sq": 0,"i_sd": 1,"i_e": 2,"epsilon": 3,"torque": 4,"omega": 5,"u": 6}
     ExtExcSyncMotor_state_space = Box(low=-1, high=1, shape=(7,), dtype=np.float64)
     assert ExtExcSyncMotor._initial_states == default_initial_state
     ExtExcSyncMotor._initial_states = new_initial_state
     assert (ExtExcSyncMotor.reset(ExtExcSyncMotor_state_space,ExtExcSyncMotor_state_positions),default_Initial_state_array)
     ExtExcSyncMotor.initializer["states"] = new_initial_state
     ExtExcSyncMotor.reset(ExtExcSyncMotor_state_space,ExtExcSyncMotor_state_positions)
     assert ExtExcSyncMotor._initial_states == new_initial_state

def test_InitPermMagSyncMotor():
     defaultPermMagSyncMotor = PermanentMagnetSynchronousMotor()
     assert defaultPermMagSyncMotor._default_motor_parameter == {"p": 3,"l_d": 0.37e-3,"l_q": 1.2e-3,"j_rotor": 0.03883,"r_s": 18e-3,"psi_p": 66e-3,}
     assert defaultPermMagSyncMotor.HAS_JACOBIAN
     assert defaultPermMagSyncMotor.motor_parameter == defaultPermMagSyncMotor._default_motor_parameter
     assert defaultPermMagSyncMotor._initial_states == defaultPermMagSyncMotor._default_initializer["states"]
     concretePermMagSyncMotor = PermanentMagnetSynchronousMotor(test_PermMagSyncMotor_parameter,None,None,test_PermMagSyncMotor_initializer)
     assert concretePermMagSyncMotor.motor_parameter == test_PermMagSyncMotor_parameter
     assert concretePermMagSyncMotor.initializer == test_PermMagSyncMotor_initializer
     assert concretePermMagSyncMotor._initial_states == test_PermMagSyncMotor_initializer["states"]
     assert defaultPermMagSyncMotor.CURRENTS_IDX == SynchronousMotor.CURRENTS_IDX
     assert defaultPermMagSyncMotor.CURRENTS == SynchronousMotor.CURRENTS

def test_PermMagSyncMotor_torque():
     defaultPermMagSyncMotor = PermanentMagnetSynchronousMotor()
     mp = defaultPermMagSyncMotor._motor_parameter
     currents = [3,1,2,3]
     expectedTorque =  (
            1.5 * mp["p"] * (mp["psi_p"] + (mp["l_d"] - mp["l_q"]) * currents[defaultPermMagSyncMotor.I_SD_IDX]) * currents[defaultPermMagSyncMotor.I_SQ_IDX]
        )
     assert defaultPermMagSyncMotor.torque(currents) == expectedTorque

def test_PermMagSyncMotor_el_Jacobian():
     defaultPermMagSyncMotor = PermanentMagnetSynchronousMotor()
     mp = defaultPermMagSyncMotor._motor_parameter
     state = [5, 5]
     omega = 60
     u_in = [50, 60]
     expectedJacobian = (
            np.array(
                [  # dx'/dx
                    [
                        -mp["r_s"] / mp["l_d"],
                        mp["l_q"] / mp["l_d"] * omega * mp["p"],
                        0,
                    ],
                    [
                        -mp["l_d"] / mp["l_q"] * omega * mp["p"],
                        -mp["r_s"] / mp["l_q"],
                        0,
                    ],
                    [0, 0, 0],
                ]
            ),
            np.array(
                [  # dx'/dw
                    mp["p"] * mp["l_q"] / mp["l_d"] * state[defaultPermMagSyncMotor.I_SQ_IDX],
                    -mp["p"] * mp["l_d"] / mp["l_q"] * state[defaultPermMagSyncMotor.I_SD_IDX] - mp["p"] * mp["psi_p"] / mp["l_q"],
                    mp["p"],
                ]
            ),
            np.array(
                [  # dT/dx
                    1.5 * mp["p"] * (mp["l_d"] - mp["l_q"]) * state[defaultPermMagSyncMotor.I_SQ_IDX],
                    1.5 * mp["p"] * (mp["psi_p"] + (mp["l_d"] - mp["l_q"]) * state[defaultPermMagSyncMotor.I_SD_IDX]),
                    0,
                ]
            ),
        )
     assert np.array_equal(expectedJacobian[0],defaultPermMagSyncMotor.electrical_jacobian(state,u_in,omega)[0])

def test_PermMagSyncMotor_el_ODE():
      defaultPermMagSyncMotor = PermanentMagnetSynchronousMotor()
      state = [5, 5]
      omega = 60
      u_in = [50, 60]
      state = [5, 5]
      #assert defaultPermMagSyncMotor.electrical_ode(state,u_in,omega) == SynchronousMotor.electrical_ode(state,u_in,omega)