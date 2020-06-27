
import os
import sys
import time

from examples.agents.simple_controllers import Controller

sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

"""Run this file from within the 'examples' folder:
>> cd examples
>> python pi_series_omega_control.py
Description:
        Environment to control a continuously controlled DC Series Motor.

        Controlled Quantity: 'omega'

        Limitations: Physical limitations of the motor like Current,Speed

        Converter : OneQuadrantConverter from converters.py
"""

if __name__ == '__main__':
    """Discrete mode: It is the mode where the input command causes the controlled variable to change
     its value in discrete steps,with a brief static condition occurring between each step.
    
    Continuous mode is the one where the input command signal causes 
    the controlled variable to remain constant,or to change in a linear(rather than step) mode. 
    """
    env = gem.make(
        'DcSeriesCont-v1',  # replace with 'DcSeriesDisc-v1' for discrete controllers
        visualization=MotorDashboard(plots=['omega' , 'torque', 'i', 'u', 'u_sup'], dark_mode=True),
        motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=5e-3, l_e=5e-3),
        load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(),
        # Pass a Class with extra parameters
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator, rg.WienerProcessReferenceGenerator(), rg.StepReferenceGenerator()
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )
    controller = Controller.make('pi_controller', env)
    # The above controller can be replaced with 'on_off'(disc),three_point(disc),'p_controller', 'cascaded pi' from simple_controllers class
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
