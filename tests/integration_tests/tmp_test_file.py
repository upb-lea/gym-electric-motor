# Following lines of code are needed to be abled to succesfully execute the import in line 7
import sys
import os
path = os.getcwd()+'/examples/classic_controllers'
sys.path.append(path)

from classic_controllers import Controller

import gym_electric_motor as gem

from gym_electric_motor.reference_generators import SinusoidalReferenceGenerator

import numpy as np


if __name__ == '__main__':

    """
    motor type:     'PermExDc'  Permanently Excited DC Motor
                    'ExtExDc'   Externally Excited MC Motor
                    'SeriesDc'  DC Series Motor
                    'ShuntDc'   DC Shunt Motor
                    
    control type:   'SC'         Speed Control
                    'TC'         Torque Control
                    'CC'         Current Control
                    
    action_type:    'Cont'      Continuous Action Space
                    'Finite'    Discrete Action Space
    """

    motor_type = 'PermExDc'
    control_type = 'SC'
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

    # definition of the reference generator

    ref_generator = SinusoidalReferenceGenerator(amplitude_range= (1,1),
                                                 frequency_range= (5,5),
                                                 offset_range = (0,0),
                                                 episode_lengths = (10001, 10001))

    # initialize the gym-electric-motor environment
    env = gem.make(motor,
                   reference_generator = ref_generator)
    
    """
        initialize the controller

        Args:
            environment                     gym-electric-motor environment
            external_ref_plots (optional)   plots of the environment, to plot all reference values
            stages (optional)               structure of the controller
            automated_gain (optional)       if True (default), the controller will be tuned automatically
            a (optional)                    tuning parameter of the symmetrical optimum (default: 4)
    
    """
    controller = Controller.make(env)
    
    state, reference = env.reset(seed=1337)

    test_states = []
    test_reference = []
    test_reward = []
    test_term = []
    test_trunc = []
    test_info = []

    # simulate the environment
    for i in range(2001):
        action = controller.control(state, reference)

        (state, reference), reward, terminated, truncated, _ = env.step(action)

        test_states.append(state)
        test_reference.append(reference)
        test_reward.append(reward)
        test_term.append(terminated)
        test_trunc.append(truncated)

        if terminated:
            env.reset()
            controller.reset()
    
    np.savez('test_data', 
             states = test_states, references = test_reference, 
             rewards = test_reward, 
             terminations = test_term, 
             truncations = test_trunc)

    env.close()

    test_data = np.load('test_data.npz')
    ref_data = np.load('./tests/integration_tests/ref_data.npz')
    
    for i in ref_data.files:

        if np.allclose(ref_data[i], test_data[i], equal_nan= True):
            print('States are equal - passed')
        else:
            print('failed')
            break

    os.remove("test_data.npz")