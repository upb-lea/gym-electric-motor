from gym_electric_motor.physical_systems.mechanical_loads import PolynomialStaticLoad
from gym_electric_motor.physical_systems.electric_motors import DcSeriesMotor
import numba
import numpy as np
import os
import sys
import time

from examples.agents.simple_controllers import Controller

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard


class NumbaDcSeriesMotor(DcSeriesMotor):

    def __init__(self, use_numba=True, **kwargs):
        super().__init__(**kwargs)
        if use_numba:
            self.make_numba()

    def make_numba(self):
        i_idx = self.I_IDX
        mp = self.motor_parameter
        l_e_prime = mp['l_e_prime']
        r_a = mp['r_a']
        r_e = mp['r_e']
        l_a = mp['l_a']
        l_e = mp['l_e']
        l_sum = l_a+l_e
        r_sum = r_a+r_e

        def numba_torque(currents):
            return l_e_prime * currents[i_idx]**2

        def numba_ode(state, u_in, omega):
            return np.array([(-r_sum * state[i_idx] - l_e_prime * omega * state[i_idx] + u_in)/l_sum])

        def numba_jac(state, u_in, omega):
            return (
                np.array([[-(r_sum + l_e_prime * omega) / l_sum]]),
                np.array([-l_e_prime * state[i_idx] / l_sum]),
                np.array([2 * l_e_prime * state[i_idx]])
            )

        self.torque = numba.jit(numba_torque, nopython=True)

        # Numba doesnt like lists and is slow when processing them. The u_in is a list. Therefore is this workaround.
        # In a Physical System refactoring this would be considered of course
        numba_ode_ = numba.jit(numba_ode, nopython=True)
        self.electrical_ode = lambda state, u_in, omega: numba_ode_(state, u_in[0], omega)
        numba_jac_ = numba.jit(numba_jac, nopython=True)
        self.electrical_jacobian = lambda state, u_in, omega: numba_jac_(state, u_in[0], omega)
        self.i_in = numba.jit(lambda currents: [currents[i_idx]], nopython=True)


class NumbaPolynomialStaticLoad(PolynomialStaticLoad):

    def __init__(self, use_numba=True, **kwargs):
        super().__init__(**kwargs)
        self.use_numba = use_numba

    def set_j_rotor(self, j_rotor):
        super().set_j_rotor(j_rotor)
        if self.use_numba:
            self._make_numba()

    def _make_numba(self):
        omega_idx = self.OMEGA_IDX
        j_rotor = self._j_total
        a = self.load_parameter['a']
        b = self.load_parameter['b']
        c = self.load_parameter['c']

        def numba_ode(t, mechanical_state, torque):
            omega = mechanical_state[omega_idx]
            sign = 1 if omega > 0 else -1 if omega < 0 else 0
            return np.array([(torque - sign * c * omega**2 - b * omega - sign * a) / j_rotor])

        def numba_jac(t, mechanical_state, torque):
            sign = 1 if mechanical_state[omega_idx] > 0 else -1 if mechanical_state[omega_idx] < 0 else 0
            return np.array([[(-b * sign - 2 * c * mechanical_state[omega_idx]) / j_rotor]]), \
                np.array([1 / j_rotor])

        self.mechanical_ode = numba.jit(numba_ode, nopython=True)
        self.mechanical_jacobian = numba.jit(numba_jac, nopython=True)


if __name__ == '__main__':

    use_numba_motor = False
    use_numba_load = False

    env = gem.make('DcSeriesCont-v1',  # replace with 'DcSeriesDisc-v1' for discrete controllers
                   state_filter=['omega', 'i'],
                   # Pass an instance
                   #visualization=MotorDashboard(plots=['i', 'omega']),
                   # Take standard class and pass parameters (Load)
                   motor=NumbaDcSeriesMotor(motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
                                            use_numba=use_numba_motor),
                   load=NumbaPolynomialStaticLoad(load_parameter=dict(a=.1, b=.01, c=0, j_load=0.001),
                                            use_numba=use_numba_load),
                   # Pass a string (with extra parameters)
                   ode_solver='scipy.solve_ivp', solver_kwargs=dict(method='LSODA'),
                   # Pass a Class with extra parameters
                   reference_generator=rg.WienerProcessReferenceGenerator())

    controller = Controller.make('pi_controller', env)

    state, reference = env.reset()
    incl_compile = time.time()
    action = controller.control(state, reference)

    (state, reference), reward, done, _ = env.step(action)
    start = time.time()
    cum_rew = 0
    for i in range(200000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)

        if done:
            env.reset()
        cum_rew += reward
    print(cum_rew)
    print('After Compile:', time.time() - start)
    print('Inclusive Compile:', time.time() - incl_compile)