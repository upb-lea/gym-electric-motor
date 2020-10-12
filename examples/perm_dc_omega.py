from examples.agents.simple_controllers import Controller
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard

'''
This example shows how GEM can be used to test discontinuous controllers.
The used three point controller is only capable of three actions: full throttle forward, full throttle backward
and idle. These actions suffice to control the speed of a permanently excited DC motor.
'''

if __name__ == '__main__':

    # define the drive environment
    env = gem.make(
        # define the permanently excited DC motor with finite-control-set
        'DcPermExDisc-v1',

        # Defines which variables to plot via the builtin dashboard monitor
        visualization=MotorDashboard(plots=['omega', 'torque', 'i', 'u', 'u_sup']),

        # Defines which numerical solver is to be used for the simulation
        ode_solver='scipy.solve_ivp',
        solver_kwargs=dict(),

        # Define a reference generator for the speed ('omega')
        # In this example, we permit random switching between sinusoidal, wiener and step reference with
        # probabilities 0.1, 0.8 and 0.1 respectively.
        # super_episode_length defines minimum and maximum time steps between switching references
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator(reference_state="omega"),
                rg.WienerProcessReferenceGenerator(reference_state="omega"),
                rg.StepReferenceGenerator(reference_state="omega")
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )

    # use a predefined controller for the environment
    # this controller reads to which state(s) the reference generator refers and
    # will be adjusted to control the corresponding state(s)
    controller = Controller.make('three_point', env)

    # reset the environment to an initial state (initial state is zero if not defined otherwise)
    state, reference = env.reset()

    cum_rew = 0
    for i in range(100000):

        # the render command updates the dashboard
        env.render()

        # the controller accepts state and reference to calculate a new action
        action = controller.control(state, reference)

        # the drive environment accepts the action and simulates until the next time step
        (state, reference), reward, done, _ = env.step(action)

        if done:
            state, reference = env.reset()
            controller.reset()
        cum_rew += reward

    print(cum_rew)
