"""Run this file from within the 'examples' folder:
>>> cd examples
>>> python pi_series_omega_control.py
"""
from examples.agents.simple_controllers import Controller
import time
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import *
from gym_electric_motor.visualization import MotorDashboard


if __name__ == '__main__':
    env = gem.make(
        'emotor-dc-series-cont-v1',
        # Pass an instance
        visualization=MotorDashboard(plotted_variables=['all'], visu_period=1),

        # Take standard class and pass parameters (Load)
        load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.05),

        # Pass a string (with extra parameters)
        ode_solver='euler', solver_kwargs={},
        # Pass a Class with extra parameters
        reference_generator=MultiReferenceGenerator(
            sub_generators=[SinusoidalReferenceGenerator, WienerProcessReferenceGenerator(), StepReferenceGenerator()],
            p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )
    controller = Controller.make('cascaded_pi', env)
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in range(100000):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)
        cum_rew += reward
    print(cum_rew)
