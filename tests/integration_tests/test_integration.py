# Following lines of code are needed to be abled to succesfully execute the import in line 7
import sys
import os
path = os.getcwd()+'/examples/classic_controllers'
sys.path.append(path)
from classic_controllers import Controller
#import pytest
import gym_electric_motor as gem

from gym_electric_motor.reference_generators import SinusoidalReferenceGenerator


import numpy as np


def simulate_env(seed = None):

    motor_type = 'PermExDc'
    control_type = 'SC'
    action_type = 'Cont'
    version = 'v0'
    
    env_id = f'{action_type}-{control_type}-{motor_type}-{version}'
   

    # definition of the reference generator

    ref_generator = SinusoidalReferenceGenerator(amplitude_range= (1,1),
                                                 frequency_range= (5,5),
                                                 offset_range = (0,0),
                                                 episode_lengths = (10001, 10001))

    # initialize the gym-electric-motor environment
    env = gem.make(env_id,
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

    (state, reference), _ = env.reset(seed)

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
    
    np.savez('./tests/integration_tests/test_data.npz', 
             states = test_states, references = test_reference, 
             rewards = test_reward, 
             terminations = test_term, 
             truncations = test_trunc)

    #env.close()

def test_simulate_env():
    simulate_env(1337)
    test_data = np.load('./tests/integration_tests/test_data.npz')
    ref_data = np.load('./tests/integration_tests/ref_data.npz')
    
    for file in ref_data.files:
        assert(np.allclose(ref_data[file], test_data[file], equal_nan= True))

    os.remove('./tests/integration_tests/test_data.npz')

    # Anti test
    simulate_env(1234)
    test_data = np.load('./tests/integration_tests/test_data.npz')

    # test only states, references and rewards
    for file in ref_data.files[0:3]:
        assert((not np.allclose(ref_data[file], test_data[file], equal_nan= True)))

    os.remove('./tests/integration_tests/test_data.npz')
   
    