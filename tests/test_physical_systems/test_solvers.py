import numpy as np
import gym_electric_motor.envs
from gym_electric_motor.physical_systems.solvers import *
import gym_electric_motor.physical_systems.solvers as pss
from ..testing_utils import DummyScipyOdeSolver
from ..conf import system, jacobian
import pytest


# region first version tests


g_initial_value = np.array([1, 6])
g_initial_time = 1
g_time_steps = [1E-5, 2E-5, 3E-5, 5E-5, 1E-4, 2E-4, 5E-4, 1E-3, 2E-3, 5E-3, 7E-3, 1E-2]


def test_euler():
    """
    tests if basic euler solver works basically
    :return:
    """
    for nsteps in [1, 5]:
        solver = EulerSolver(nsteps)
        integration_testing(solver)


@pytest.mark.parametrize("integrator", ['dopri5', 'dop853'])
# 'vode', 'zvode', 'lsoda', could be added, but does not work due to wrong integration times
def test_scipyode(integrator):
    """
    test scipy.ode integrators
    :param integrator: chosen integrator (string)
    :return:
    """
    for nsteps in [1, 5]:
        kwargs = {'nsteps': nsteps, }
        solver = ScipyOdeSolver(integrator, **kwargs)
        integration_testing(solver)


@pytest.mark.parametrize("integrator", ['RK45', 'RK23', 'Radau', 'BDF', 'LSODA'])
def test_scipy_ivp(integrator):
    """
    test scipy.solveivp integrator
    :param integrator: chosen integrator (string)
    :return:
    """
    solver = ScipySolveIvpSolver(method=integrator)
    integration_testing(solver)


def test_scipyodeint():
    """
    test scipy.odeint integrator
    :return:
    """
    for nsteps in [1, 5]:
        kwargs = {'nsteps': nsteps, }
        solver = ScipyOdeIntSolver()
        integration_testing(solver)


def integration_testing(solver):
    """
    tests if no errors are raised due to the integration
    :param solver: initialized integrator
    :return:
    """
    time_step = [g_initial_time + step for step in g_time_steps]
    u = 0.0
    system_equation = lambda t, state, u: system(t, state, u)
    system_jacobian = lambda t, state, u: jacobian(t, state, u)
    solver.set_system_equation(system_equation, system_jacobian)
    solver.set_initial_value(g_initial_value, g_initial_time)
    assert g_initial_time == solver.t
    for steps in time_step:
        solver.set_f_params(u)
        state = solver.integrate(steps)
        assert all(state == solver.y), "state error"
        assert solver.t == steps, "time error, desired and actual time are not the same"


def test_compare_solver():
    """
    Compare different solver
    Basic euler solver
    solve.ode runge kutta
    Solveivp runge kutta
    solve.odeint
    :return:
    """
    # initialize all solver
    solver_1 = EulerSolver()
    solver_2 = ScipyOdeSolver('dopri5')
    solver_3 = ScipySolveIvpSolver(method='Radau')
    solver_4 = ScipyOdeIntSolver()
    solver = [solver_1, solver_2, solver_3, solver_4]
    # define the integration steps
    time_step = [g_initial_time + step for step in g_time_steps]
    u = 0.0  # initial input
    # set the system equation and jacobian
    system_equation = lambda t, state, u: system(t, state, u)
    system_jacobian = lambda t, state, u: jacobian(t, state, u)
    [solve.set_system_equation(system_equation, system_jacobian) for solve in solver]
    # set initial values of the solver
    [solve.set_initial_value(g_initial_value, g_initial_time) for solve in solver]
    # integrate for given steps
    for steps in time_step:
        u += 0.8  # increase the input in each step
        [solve.set_f_params(u) for solve in solver]
        [solve.integrate(steps) for solve in solver]
        # test if all solver integrated for the same time
        assert solver_1.t == solver_2.t == solver_3.t == solver_4.t
        # test if the relative difference between the states is small
        abs_values = [sum(abs(solver_1.y)), sum(abs(solver_2.y)), sum(abs(solver_3.y)), sum(abs(solver_4.y))]
        assert max(abs_values) / min(abs_values) - 1 < 1E-4, "Time step at the error: " + str(steps)


# endregion


# region second version tests

class TestOdeSolver:
    """
    class for testing OdeSolver
    """
    # defined test values
    _initial_time = 0.1
    _initial_state = np.array([15, 0.23, 0.35])

    def test_set_initial_value(self):
        """
        test set_initial_value()
        :return:
        """
        # setup test scenario
        test_object = OdeSolver()
        # call function to test
        test_object.set_initial_value(self._initial_state, self._initial_time)
        # verify the expected results
        assert all(test_object._y == test_object.y) and all(test_object._y == self._initial_state), 'unexpected state'
        assert test_object.t == test_object._t == self._initial_time, 'unexpected time'

    @pytest.mark.parametrize('jacobian_', [jacobian, None])
    def test_set_system_equation(self, jacobian_):
        """
        test set_system_equation()
        :param jacobian_: jacobian of the test system
        :return:
        """
        # setup test scenario
        test_object = OdeSolver()
        # call function to test
        test_object.set_system_equation(system, jacobian_)
        # verify the expected results
        assert test_object._system_equation == system, 'system equation is not passed correctly'
        assert test_object._system_jacobian == jacobian_, 'jacobian is not passed correctly'

    @pytest.mark.parametrize('args', [[], ['nsteps', 5], 42])
    def test_set_f_params(self, args):
        """
        test set_f_params()
        :param args: arguments that should be tested
        :return:
        """
        # setup test scenario
        test_object = OdeSolver()
        # call function to test
        test_object.set_f_params(args)
        # verify the expected results
        assert test_object._f_params[0] == args, 'arguments are not passed correctly for the system'


class TestEulerSolver:
    """
    class for testing EulerSolver
    """
    # defined test values
    _t = 5e-4
    _state = np.array([1, 6])

    def monkey_integrate(self, t):
        """
        mock function for _integrate()
        :param t: time
        :return:
        """
        assert t == self._t, 'unexpected time at the end of the integration'
        return self._state

    @pytest.mark.parametrize('nsteps, expected_integrate', [(1, 0), (5, 1)])
    def test_init(self, nsteps, expected_integrate):
        """
        test initialization of EulerSolver
        :param nsteps: number of steps
        :param expected_integrate: expected result for integration function
        :return:
        """
        # call function to test
        test_object = EulerSolver(nsteps)
        # verify the expected results
        integration_functions = [test_object._integrate_one_step, test_object._integrate_nsteps]
        assert test_object._nsteps == nsteps, 'nsteps argument not passed correctly'
        assert test_object._integrate == integration_functions[expected_integrate], 'unexpected integrate function'

    def test_integrate(self, monkeypatch):
        """
        test integrate()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = EulerSolver()
        monkeypatch.setattr(test_object, '_integrate', self.monkey_integrate)
        # call function to test
        state = test_object.integrate(self._t)
        # verify the expected results
        assert all(state == self._state), 'unexpected state after integration'

    @pytest.mark.parametrize('nsteps, expected_state',
                             [(1, np.array([1.006, 6.0258])), (3, np.array([1.00596811, 6.025793824]))])
    def test_private_integration(self, nsteps, expected_state):
        """
        test _integration() for different cases
        :param nsteps: number of steps of integration
        :param expected_state: expected resulting state
        :return:
        """
        # setup test scenario
        test_object = EulerSolver(nsteps=nsteps)
        test_object.set_system_equation(system, jacobian)
        tau = 1e-3
        u = 2
        test_object.set_initial_value(self._state, self._t)
        test_object.set_f_params(u)
        # call function to test
        state = test_object.integrate(self._t + tau)
        # verify the expected results
        assert sum(abs(state - expected_state)) < 1E-6, 'unexpected state after integration'


class TestScipyOdeSolver:
    """
    class for testing ScipyOdeSolver
    """
    # defined test values
    _kwargs = {'nsteps': 5}
    _integrator = 'dop853'
    _args = 2
    _tau = 1e-3
    _initial_time = 0.1
    _initial_state = np.array([1, 6])

    # counter
    _monkey_super_set_system_equation_counter = 0
    _monkey_set_params_counter = 0

    def monkey_set_integrator(self, integrator, **kwargs):
        """
        mock function for set_integrator()
        :param integrator:
        :param kwargs:
        :return:
        """
        assert integrator == self._integrator, 'integrator not passed correctly'
        assert kwargs == self._kwargs, 'unexpected additional arguments. Keep in mind None and {}.'

    def monkey_super_set_system_equation(self, system_equation, jacobian_):
        """
        mock function for super().set_system_equation()
        :param system_equation: function of system equation
        :param jacobian_: function of jacobian
        :return:
        """
        self._monkey_super_set_system_equation_counter += 1
        assert system_equation == system, 'unexpected system equation'
        assert jacobian_ == jacobian, 'unexpected jacobian'

    def monkey_set_params(self, **args):
        """
        mock function for set params
        :param args:
        :return:
        """
        self._monkey_set_params_counter += 1
        assert self._args == (args,), 'unexpected additional arguments. Keep the type in mind'

    def test_init(self):
        """
        test initialization of ScipyOdeSolver
        :return:
        """
        # call function to test
        test_object = ScipyOdeSolver(integrator=self._integrator, **self._kwargs)
        assert test_object._solver is None
        assert test_object._solver_args == self._kwargs, 'unexpected additional arguments. Keep in mind None and {}.'
        assert test_object._integrator == self._integrator, 'unexpected initialization of integrate function'

    def test_set_system_equation(self, monkeypatch):
        """
        test set_system_equation()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'ode', DummyScipyOdeSolver)
        test_object = ScipyOdeSolver(integrator=self._integrator, **self._kwargs)
        monkeypatch.setattr(OdeSolver, 'set_system_equation', self.monkey_super_set_system_equation)
        # call function to test
        test_object.set_system_equation(system, jacobian)
        # verify the expected results
        assert isinstance(test_object._ode, DummyScipyOdeSolver), 'the ode is no DummyScipyOdeSolver'
        assert self._monkey_super_set_system_equation_counter == 1, 'super().set_system_equation() is not called once'
        assert test_object._ode._set_integrator_counter == 1, 'set_integrator() is not called once'

    def test_set_initial_value(self, monkeypatch):
        """
        test set_initial_value()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'ode', DummyScipyOdeSolver)
        test_object = ScipyOdeSolver(integrator=self._integrator, **self._kwargs)
        test_object.set_system_equation(system, jacobian)
        # call function to test
        test_object.set_initial_value(self._initial_state, self._initial_time)
        # verify the expected results
        assert test_object._ode._set_initial_value_counter == 1, 'set_initial_value() is not called once'

    def test_set_f_params(self, monkeypatch):
        """
        test set_f_params()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'ode', DummyScipyOdeSolver)
        test_object = ScipyOdeSolver(integrator=self._integrator, **self._kwargs)
        test_object.set_system_equation(system, jacobian)
        # call function to test
        test_object.set_f_params(self._args)
        # verify the expected results
        assert test_object._ode._set_f_params_counter == 1, 'set_f_params() is not called once'

    def test_integrate(self, monkeypatch):
        """
        test integrate()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'ode', DummyScipyOdeSolver)
        test_object = ScipyOdeSolver(integrator=self._integrator, **self._kwargs)
        test_object.set_system_equation(system, jacobian)
        test_object.set_initial_value(self._initial_state, self._initial_time)
        test_object.set_f_params(self._args)
        # call function to test
        result = test_object.integrate(self._tau + self._initial_time)
        # verify the expected results
        assert all(result == self._initial_state * 2), 'unexpected result of the integration'
        assert test_object._ode._integrate_counter == 1, '_ode._integrate() is not called once()'


class TestScipySolveIvpSolver:
    """
    class for testing ScipySolveIvpSolver
    """
    # defined test values
    _state = np.array([1, 6])
    _tau = 1e-3
    args = 2

    # counter
    _monkey_set_f_params_counter = 0
    _monkey_super_set_system_equation_counter = 0

    def monkey_super_set_system_equation(self, system_equation, jac):
        """
        mock function for super().set_system_equation()
        :return:
        """
        self._monkey_super_set_system_equation_counter += 1
        assert system_equation == system, 'unexpected system passed'
        assert jac == jacobian, 'unexpected jacobian passed'

    def monkey_set_f_params(self):
        """
        mock function for set_f_params()
        :return:
        """
        self._monkey_set_f_params_counter += 1

    def monkey_solve_ivp(self, function, time, state, t_eval, **kwargs):
        """
        mock function for solve_ivp()
        all arguments are similar to that function
        :param function:
        :param time:
        :param state:
        :param t_eval:
        :param jac:
        :param kwargs:
        :return:
        """
        assert time == [0.0, self._tau], 'unexpected time passed'
        assert all(state == self._state), 'unexpected state passed'
        assert t_eval == [self._tau], 'unexpected time for evaluation'

        class Result:
            """
            simple class necessary for correct testing
            """
            y = np.array([[7, 1], [4, 6]])

        return Result()

    def test_init(self):
        """
        test initialization of ScipySolveIvpSolver
        :return:
        """
        # setup test scenario
        _kwargs = {'nsteps': 5}
        # call function to test
        test_object = ScipySolveIvpSolver(**_kwargs)
        # verify the expected results
        assert test_object._solver_kwargs == _kwargs, 'unexpected additional arguments. Keep in mind None and {}'

    def test_set_system_equation(self, monkeypatch):
        """
        test set_system_equation()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(OdeSolver, 'set_system_equation', self.monkey_super_set_system_equation)
        monkeypatch.setattr(ScipySolveIvpSolver, 'set_f_params', self.monkey_set_f_params)
        test_object = ScipySolveIvpSolver()
        # call function to test
        test_object.set_system_equation(system, jacobian)

    def test_set_f_params(self):
        """
        test set_f_params()
        :return:
        """
        # setup test scenario
        _expected_result = np.array([6, 25.8])
        _state = np.array([1, 6])
        _tau = 1e-3
        args = 2
        test_object = ScipySolveIvpSolver()
        test_object.set_system_equation(system, jacobian)
        # call function to test
        test_object.set_f_params(args)
        # verify the expected results
        assert sum(abs(test_object._system_equation(_tau, _state, args) - _expected_result)) < 1E-6,\
            'unexpected result after set_f_params()'

    def test_integrate(self, monkeypatch):
        """
        test integrate()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'solve_ivp', self.monkey_solve_ivp)
        test_object = ScipySolveIvpSolver()
        test_object.set_system_equation(system, jacobian)
        test_object.set_initial_value(self._state, 0.0)
        test_object.set_f_params(self.args)
        # call function to test
        result = test_object.integrate(self._tau)
        # verify the expected results
        assert all(result == self._state), 'unexpected state after integration'


class TestScipyOdeIntSolver:
    """
    class for testing ScipyOdeIntSolver
    """
    # defined test values
    _state = np.array([1, 6])
    _tau = 1e-3
    args = 2

    # counter
    _monkey_set_f_params_counter = 0
    _monkey_super_set_system_equation_counter = 0

    def monkey_super_set_system_equation(self, system_equation, jac):
        """
        mock function for super().set_system_equation()
        :param system_equation:
        :param jac:
        :return:
        """
        self._monkey_super_set_system_equation_counter += 1
        assert system_equation == system, 'unexpected system passed'
        assert jac == jacobian, 'unexpected jacobian passed'

    def monkey_set_f_params(self):
        """
        mock function for set_f_params()
        :return:
        """
        self._monkey_set_f_params_counter += 1

    def monkey_ode_int(self, function, y, time, args, Dfun, tfirst, **kwargs):
        """
        mock function for odeint
        all parameters are the same as in odeint
        :param function:
        :param y:
        :param time:
        :param args:
        :param Dfun:
        :param tfirst:
        :param kwargs:
        :return:
        """
        assert all(y == self._state), 'unexpected state before integration'
        assert time == [0, self._tau], 'unexpected time steps for integration'
        assert args == (self.args,), 'unexpected arguments'
        assert tfirst
        return np.array([[0, 2], self._state])

    def test_init(self):
        """
        test initialization of ScipyOdeIntSolver
        :return:
        """
        # setup test scenario
        _kwargs = {'nsteps': 5}
        # call function to test
        test_object = ScipyOdeIntSolver(**_kwargs)
        # verify the expected results
        assert test_object._solver_args == _kwargs, 'unexpected additional arguments. Keep the type in mind.'

    def test_set_system_equation(self, monkeypatch):
        """
        test set_system_equation()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(OdeSolver, 'set_system_equation', self.monkey_super_set_system_equation)
        monkeypatch.setattr(ScipyOdeIntSolver, 'set_f_params', self.monkey_set_f_params)
        test_object = ScipyOdeIntSolver()
        # call function to test
        test_object.set_system_equation(system, jacobian)

    def test_integrate(self, monkeypatch):
        """
        test integrate()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(pss, 'odeint', self.monkey_ode_int)
        test_object = ScipyOdeIntSolver()
        test_object.set_system_equation(system, jacobian)
        test_object.set_initial_value(self._state, 0.0)
        test_object.set_f_params(self.args)
        # call function to test
        result = test_object.integrate(self._tau)
        # verify the expected results
        assert all(result == self._state), 'unexpected result after integration'


# endregion
