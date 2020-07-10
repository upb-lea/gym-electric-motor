from agents.simple_controllers import Controller
import time
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor import reference_generators as rg
from gym_electric_motor.visualization import MotorDashboard
from tqdm import tqdm
import numpy as np
from gym.spaces import Box
from gym_electric_motor.reward_functions.constraint_monitor import ConstraintMonitor

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

# constraints, as gym.space (when bounds aren't symmetrical) or np.ndarray
constraints_1 = Box(low=-0.9, high=0.9, shape=(5,))
constraints_2 = Box(low=np.array([-70, -440, -70, -300, -420]),
                    high=np.array([70, 440, 70, 300, 420]))
# for 70% of normal limits
discount_factor = 0.7

ConMon = ConstraintMonitor(constraints=constraints_2)
#ConMon = ConstraintMonitor(discount_factor=discount_factor)

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
        constraint_monitor=ConMon,
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