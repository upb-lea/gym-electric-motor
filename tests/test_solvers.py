import numpy as np
import gym_electric_motor.envs
from gym_electric_motor.physical_systems.solvers import *
import pytest


"""
    simulate the system
    d/dt[x,y]=[[3 * x + 5 * y - 2 * x * y + 3 * x**2 - 0.5 * y**2],
    [10 - 0.6 * x + 0.9 * y**2 - 3 * x**2 *y]]
    with the initial value [1, 6]
"""


g_initial_value = np.array([1, 6])
g_initial_time = 1
g_time_steps = [1E-5, 2E-5, 3E-5, 5E-5, 1E-4, 2E-4, 5E-4, 1E-3, 2E-3, 5E-3, 7E-3, 1E-2]


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
                       10 - 0.6 * x + 0.9 * y ** 2 - 3 * x**2 * y + u])
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
    result = np.array([[3-2*y+6*x, 5-2*x-y], [-0.6-6*x*y, .8*y-3*x**2]])
    return result


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
    solver.set_system_equation(system_equation)
    solver.set_j_params(system_jacobian)
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
    solver_2 = ScipyOdeSolver('dopri')
    solver_3 = ScipySolveIvpSolver(method='RK45')
    solver_4 = ScipyOdeIntSolver()
    solver = [solver_1, solver_2, solver_3, solver_4]
    # define the integration steps
    time_step = [g_initial_time + step for step in g_time_steps]
    u = 0.0 # initial input
    # set the system equation and jacobian
    system_equation = lambda t, state, u: system(t, state, u)
    system_jacobian = lambda t, state, u: jacobian(t, state, u)
    [solve.set_system_equation(system_equation) for solve in solver]
    [solve.set_j_params(system_jacobian) for solve in solver]
    # set initial values of the solver
    [solve.set_initial_value(g_initial_value, g_initial_time) for solve in solver]
    # integrate for given steps
    for steps in time_step:
        u += 0.8 # increase the input in each step
        [solve.set_f_params(u) for solve in solver]
        [solve.integrate(steps) for solve in solver]
        # test if all solver integrated for the same time
        assert solver_1.t == solver_2.t == solver_3.t == solver_4.t
        # test if the relative difference between the states is small
        abs_values = [sum(abs(solver_1.y)), sum(abs(solver_2.y)), sum(abs(solver_3.y)), sum(abs(solver_4.y))]
        assert max(abs_values)/min(abs_values)-1 < 1E-4, "Time step at the error: " + str(steps)


