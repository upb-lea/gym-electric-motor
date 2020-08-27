"""Run this file from within the 'examples' folder:
>> cd examples
>> python pi_series_omega_control.py
"""
from agents.simple_controllers import Controller
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

# This example shows the behavior of GEM when using standard linear controllers.
# In the following, we use a PI controller to control the speed of a series DC motor.

if __name__ == '__main__':

    # Define the drive environment
    env = gem.make(
        # Define the series DC motor with continuous-control-set
        'DcSeriesCont-v1',

        # Set the electric parameters of the motor
        motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),

        # Set the parameters of the mechanical polynomial load (the default load class)
        load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),

        # Define a reference generator for the speed ('omega')
        # In this example, we permit random switching between sinusoidal, wiener and step reference with
        # probabilities 0.1, 0.8 and 0.1 respectively.
        # super_episode_length defines minimum and maximum time steps between switching references
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator(reference_state='omega'),
                rg.WienerProcessReferenceGenerator(reference_state='omega'),
                rg.StepReferenceGenerator(reference_state='omega')
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        ),

        # Defines which variables to plot via the builtin dashboard monitor
        visualization=MotorDashboard(plots=['omega', 'i', 'reward', 'action_0', 'mean_reward'], dark_mode=False),

        # Defines which numerical solver is to be used for the simulation
        ode_solver='scipy.solve_ivp',
        solver_kwargs=dict(),
    )

    # use a predefined controller for the environment
    # this controller reads to which state(s) the reference generator refers and
    # will be adjusted to control the corresponding state(s)
    controller = Controller.make('pi_controller', env)

    # reset the environment to an initial state (initial state is zero if not defined otherwise)
    state, reference = env.reset()

    # get the system time to measure program duration
    start = time.time()

    cum_rew = 0
    for i in range(100000):

        # the render command updates the dashboard
        env.render()

        # the controller accepts state and reference to calculate a new action
        action = controller.control(state, reference)

        # the drive environment accepts the action and simulates until the next time step
        (state, reference), reward, done, _ = env.step(action)

        if done:
            env.reset()
            controller.reset()
        cum_rew += reward

    print(cum_rew)
