from agents.simple_controllers import Controller
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from tqdm import tqdm
import numpy as np
import gym
from gym.spaces import Box
from gym_electric_motor.constraint_monitor import ConstraintMonitor

rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator, rg.WienerProcessReferenceGenerator(), rg.StepReferenceGenerator()
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )

const_sub_gen = [rg.ConstReferenceGenerator(reference_value=0.20),
                 rg.ConstReferenceGenerator(reference_value=0.25),
                 rg.ConstReferenceGenerator(reference_value=0.30),
                 rg.ConstReferenceGenerator(reference_value=0.35),
                 rg.ConstReferenceGenerator(reference_value=0.40)]


const_switch_gen = rg.SwitchedReferenceGenerator(const_sub_gen,
                                                 super_episode_length=(8000, 12000))


# the following external monitors are examples for the general use-case.
class ExternalMonitorClass:
    """
    A Class for defining a ConstraintMonitor. The general structure of the
    class is arbitrary, but a __call__() method, with a return-value out of
    [0, 1] is necessary. The return indicates the violation of the
    constraint-conditions
    """
    def __init__(self):
        self._constraints = Box(low=np.array([-70, -440, -70, -300, -420]),
                                high=np.array([70, 440, 70, 300, 420]))

    def __call__(self, state, observed_states, **kwargs):

        return self.check_violation(state, observed_states)

    def check_violation(self, state, observed_states):
        lower = self._constraints.low
        upper = self._constraints.high
        return float(
            (abs(state) < lower).any() or (abs(state) > upper).any())


def external_monitor_func(state, physical_system, k, **kwargs):
    """
    external monitor function with a return-value out of [0, 1], indicating the
    violation of the constraint conditions. Additional function parameters
    can be given to the ConstraintMonitor separately
    """
    factor = 0.9
    limits = physical_system.limits * factor
    check = float((abs(state) > abs(limits)).any())
    return check


# for 70% of normal limits
discount_factor = 0.7

if __name__ == '__main__':
    env = gem.make(
        'DcSeriesCont-v1',
        # visualization=MotorDashboard(plotted_variables='all', visu_period=1),
        visualization=MotorDashboard(plots=['omega', 'reward', 'i'],
                                     dark_mode=True),
        motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
        # Take standard class and pass parameters (Load)
        reward_power=0.5,
        # choose constraint monitor
        #constraint_monitor=ExternalMonitorClass(),
        constraint_monitor=external_monitor_func,
        load_parameter=dict(a=0.01, b=.1, c=0.1, j_load=.06),
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp',
        # Pass a Class with extra parameters
        reference_generator=rg.SwitchedReferenceGenerator(
            sub_generators=[
                rg.SinusoidalReferenceGenerator,
                rg.WienerProcessReferenceGenerator(),
                rg.StepReferenceGenerator()
            ], p=[0.1, 0.8, 0.1], super_episode_length=(1000, 10000)
        )
    )

    controller = Controller.make('pi_controller', env)
    state, reference = env.reset()
    start = time.time()
    cum_rew = 0
    for i in tqdm(range(50000)):
        env.render()
        action = controller.control(state, reference)
        (state, reference), reward, done, _ = env.step(action)

        if done:
            env.reset()
        cum_rew += reward
    env.close()

    print(cum_rew)