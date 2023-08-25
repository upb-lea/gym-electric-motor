import gymnasium as gym
import numpy as np
from classic_controllers import Controller
from externally_referenced_state_plot import ExternallyReferencedStatePlot
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gymnasium.vector import make as make_vec_env
import multiprocessing
import pydevd 

def make_gem_env(motor, num_envs ,visualization, render_mode):
    return gem.make(motor, num_envs ,visualization, render_mode)


if __name__ == '__main__':
    
    motor_type = 'PermExDc'
    control_type = 'TC'
    action_type = 'Cont'

    motor = action_type + '-' + control_type + '-' + motor_type + '-v0'

    if motor_type in ['PermExDc', 'SeriesDc']:
        states = ['omega', 'torque', 'i', 'u']
    elif motor_type == 'ShuntDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u']
    elif motor_type == 'ExtExDc':
        states = ['omega', 'torque', 'i_a', 'i_e', 'u_a', 'u_e']
    else:
        raise KeyError(motor_type + ' is not available')

    external_ref_plots = [ExternallyReferencedStatePlot(state) for state in states]

        # initialize the gym-electric-motor environment
    env = make_vec_env(motor, num_envs = 2 , env_fn = make_gem_env(motor, 2,visualization=MotorDashboard(additional_plots=external_ref_plots), render_mode="figure_once"))

    state, reference = env.reset(seed=42)

    actions = np.array([1, 0])

    env_buff = gem.make(motor, visualization=MotorDashboard(additional_plots=external_ref_plots), render_mode="figure")
    controller1 = Controller.make(env_buff, external_ref_plots=external_ref_plots)
    controller2 = Controller.make(env_buff, external_ref_plots=external_ref_plots)

    for i in range(10001):
        actions = [controller1.control(state[0][0], state[1][0]),controller2.control(state[0][1], state[1][1])] 
        observations, rewards, termination, truncation, infos = env.step(actions)

    env.close()

