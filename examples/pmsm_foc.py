import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator, \
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from examples.agents.simple_controllers import Controller
import time
import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))
if __name__ == '__main__':

    # Changing i_q reference and constant 0 i_d reference.
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
    d_generator = ConstReferenceGenerator('i_sd', 0)
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    # Change of the default motor parameters.
    motor_parameter = dict(
        r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06
    )
    limit_values = dict(
        i=160 * 1.41,
        omega=12000 * np.pi / 30,
        u=450
    )
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    # Create the environment
    env = gem.make(
        'emotor-pmsm-cont-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard(plots=['i_sq', 'i_sd', 'u_sd', 'u_sq']), visu_period=1,
        load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
        control_space='dq',
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp', solver_kwargs={},
        # Pass an instance
        reference_generator=rg,
        u_sup=400,
        motor_parameter=motor_parameter,
        limit_values=limit_values,
        nominal_values=nominal_values,
        state_filter=['i_sq', 'i_sd', 'epsilon']
    )

    controller = Controller.make('foc_controller', env)
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(100000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        if done:
            env.reset()
        cum_rew += reward
    print(cum_rew)
