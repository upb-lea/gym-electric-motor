import numpy as np
from ..testing_utils import DummyConverter, DummyLoad, DummyNoise, DummyOdeSolver, DummyVoltageSupply, DummyElectricMotor,\
    mock_instantiate, instantiate_dict
from gym_electric_motor.physical_systems import physical_systems as ps, converters as cv, electric_motors as em,\
    mechanical_loads as ml, voltage_supplies as vs, solvers as sv
from gym.spaces import Box
import pytest


class TestSCMLSystem:
    """
    Base Class to test all PhysicalSystems that derive from SCMLSystem
    """
    class_to_test = ps.SCMLSystem

    def mock_build_state(self, motor_state, torque, u_in, u_sup):
        """Function to mock an arbitrary build_state function to test the SCMLSystem
        """
        self.motor_state = motor_state
        self.torque = torque
        self.u_in = u_in
        self.u_sup = u_sup
        return np.concatenate((
                self.motor_state[:len(DummyLoad.state_names)], [torque],
                self.motor_state[len(DummyLoad.state_names):], [u_sup]
        ))

    @pytest.fixture
    def scml_system(self, monkeypatch):
        """
        Returns an instantiated SCMLSystem with Dummy Components and mocked abstract functions
        """
        monkeypatch.setattr(
            self.class_to_test,
            '_build_state_names',
            lambda _:
            DummyLoad.state_names + ['torque'] + DummyElectricMotor.CURRENTS + DummyElectricMotor.VOLTAGES + ['u_sup']
        )
        monkeypatch.setattr(
            self.class_to_test,
            '_build_state_space',
            lambda _, state_names: Box(
                low=np.zeros_like(state_names, dtype=float),
                high=np.zeros_like(state_names, dtype=float)
            )
        )
        return self.class_to_test(
            converter=DummyConverter(),
            motor=DummyElectricMotor(),
            load=DummyLoad(),
            supply=DummyVoltageSupply(),
            ode_solver=DummyOdeSolver(),
            noise_generator=DummyNoise()
        )

    def test_reset(self, scml_system):
        """Test the reset function in the physical system"""
        scml_system._t = 12
        scml_system._k = 33
        state_space = scml_system.state_space
        state_positions = scml_system.state_positions
        initial_state = scml_system.reset()
        target = (np.array([0, 0, 0, 0, 0, 0, 560]) + scml_system._noise_generator.reset()) / scml_system.limits
        assert np.all(initial_state == target), 'Initial states of the system are incorrect'
        assert scml_system._t == 0, 'Time of the system was not set to zero after reset'
        assert scml_system._k == 0, 'Episode step of the system was not set to zero after reset'
        assert scml_system.converter.reset_counter == scml_system.electrical_motor.reset_counter \
            == scml_system.mechanical_load.reset_counter == scml_system.supply.reset_counter,\
            'The reset was not passed to all components of the SCMLSystem'
        assert scml_system._ode_solver.t == 0, 'The ode solver was not reset correctly'
        assert all(scml_system._ode_solver.y == np.zeros_like(
            scml_system.mechanical_load.state_names + scml_system.electrical_motor.CURRENTS, dtype=float
        )), ' The ode solver was not reset correctly'

    def test_system_equation(self, scml_system):
        """Tests the system equation function"""
        state = np.random.rand(4)
        currents = state[[2, 3]]
        torque = scml_system.electrical_motor.torque(currents)
        u_in = np.random.rand(2)
        t = np.random.rand()
        derivative = scml_system._system_equation(t, state, u_in)
        assert all(
            derivative == np.array([torque, -torque, currents[0] - u_in[0], currents[1] - u_in[1]])
        ), 'The system equation return differs from the expected'
        assert scml_system.mechanical_load.t == t, 'The time t was not passed through to the mech. load equation'
        assert np.all(scml_system.mechanical_load.mechanical_state == state[:2]),\
            'The mech. state was not returned correctly'

    def test_simulate(self, scml_system):
        """Test the simulation function of the SCMLSystem"""

        # Reset the system and take a random action
        scml_system.reset()
        action = scml_system.action_space.sample()
        # Set a defined intitial state
        ode_state = np.array([3, 4, 5, 6])
        scml_system._ode_solver.set_initial_value(ode_state)
        # Perform the action on the system
        next_state = scml_system.simulate(action)
        solver_state_me = scml_system._ode_solver.y[:len(DummyLoad.state_names)]
        solver_state_el = scml_system._ode_solver.y[len(DummyLoad.state_names):]
        torque = [scml_system.electrical_motor.torque(solver_state_el)]
        u_sup = [scml_system.supply.u_nominal]
        u_in = [u * u_sup[0] for u in scml_system.converter.u_in]
        # Calculate the next state
        desired_next_state = (
            np.concatenate((solver_state_me, torque, solver_state_el, u_in, u_sup))
            + scml_system._noise_generator.noise()
        ) / scml_system.limits

        # Assertions for correct simulation
        assert all(desired_next_state == next_state), 'The calculated next state differs from the expected one'
        assert scml_system.converter.action == action, 'The action was not passed correctly to the converter'
        assert scml_system.converter.action_set_time == 0, 'The action start time was passed incorrect to the converter'
        assert scml_system.converter.last_i_out == scml_system.electrical_motor.i_in(scml_system._ode_solver.last_y[2:])


    def test_system_jacobian(self, scml_system):
        """Tests for the system jacobian function"""
        el_jac = np.arange(4).reshape(2, 2)
        el_over_omega = np.arange(4, 6)
        torque_over_el = np.arange(6, 8)
        # Set the el. jacobian returns to specified values
        scml_system.electrical_motor.electrical_jac_return = (el_jac, el_over_omega, torque_over_el)
        me_jac = np.arange(8, 12).reshape(2, 2)
        me_over_torque = np.arange(12, 14)
        # Set the mech. jabobian returns to specified values
        scml_system.mechanical_load.mechanical_jac_return = me_jac, me_over_torque
        sys_jac = scml_system._system_jacobian(0, np.array([0, 1, 2, 3]), [0, -1])

        #
        assert np.all(sys_jac[-2:, -2:] == el_jac), 'The el. jacobian is false'
        assert np.all(sys_jac[:2, :2] == me_jac), 'The mech. jacobian is false'
        assert np.all(sys_jac[2:, 0] == el_over_omega), 'the derivative of the el.state over omega is false'
        assert np.all(sys_jac[2:, 1] == np.zeros(2))
        assert np.all(sys_jac[:-2, 2:] == np.array([[72, 84], [78, 91]])), 'The derivative of the mech.state ' \
                                                                           'over the currents is false'
