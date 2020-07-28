import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import gym_electric_motor as gem
from gym_electric_motor.physical_systems.converters import DiscMultiConverter, DiscFourQuadrantConverter, DiscTwoQuadrantConverter
from gym_electric_motor.visualization import MotorDashboard, ConsolePrinter
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation

"""
This example is based on stable baselines3 0.8.0a5. Since it is still being frequently updated 
some parts of this code might be broken or not the latest recommended way of working with it
"""

class SqdCurrentMonitor:
    """
    monitor for squared currents:
    
    i_sd**2 + i_sq**2 < 1.5 * nominal_limit 
    """
    def __init__(self, ):
        self.I_SD_IDX = 0
        self.I_SQ_IDX = 1
        self.EPSILON_IDX = 2
        self.CURRENTS_IDX = [0, 1]
        self.CURRENTS = ['i_sd', 'i_sq']
        self.VOLTAGES = ['u_sd', 'u_sq']
    
    def __call__(self, state, observed_states, k, physical_system):
        # normalize to limit_values, since state is normalized
        nominal_values = physical_system.nominal_state / abs(physical_system.limits)
        limits = 1.5 * nominal_values
        # calculating squared currents as observed measure 
        sqd_currents = state[self.I_SD_IDX]**2 + state[self.I_SQ_IDX]**2
        
        return any(sqd_currents > limits)
        

motor_parameter = dict(p=3,  # [p] = 1, nb of pole pairs
                       r_s=17.932e-3,  # [r_s] = Ohm, stator resistance
                       l_d=0.37e-3,  # [l_d] = H, d-axis inductance
                       l_q=1.2e-3,  # [l_q] = H, q-axis inductance
                       psi_p=65.65e-3,  # [psi_p] = Vs, magnetic flux of the permanent magnet
                       )  # BRUSA
u_sup = 350
nominal_values=dict(omega=4000*2*np.pi/60,
                    i=230,
                    u=u_sup
                    )

limit_values=nominal_values.copy()

q_generator = gem.reference_generators.WienerProcessReferenceGenerator(reference_state='i_sq')
d_generator = gem.reference_generators.WienerProcessReferenceGenerator(reference_state='i_sd')
rg = gem.reference_generators.MultipleReferenceGenerator([q_generator, d_generator])

env = gem.make(# define a PMSM with continuous action space
               "PMSMDisc-v1",
               
               visualization = ConsolePrinter(verbose = 1),
               #MotorDashboard(plots = ['i_sq', 'i_sd']),
    
               # parameterize the PMSM
               motor_parameter=motor_parameter,
               #converter = converter,
    
               # update the limitations of the state space
               limit_values=limit_values,
               nominal_values=nominal_values,
               
               # define the DC link voltage
               u_sup=u_sup, 
               
               # define the speed at which the motor is operated
               #load=gem.physical_systems.ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30), 
               
               # define the duration of one sampling step
               tau=100e-6, 
               
               # turn off terminations via limit violation and parameterize the reward function
               reward_function=gem.reward_functions.WeightedSumOfErrors(observed_states='currents', 
                                                                        constraint_monitor = SqdCurrentMonitor(),
                                                                        #reward_weights={'i_sq': 1, 'i_sd': 1},
                                                                        reward_power=1
                                                                       ),
               # define the reference generator
               reference_generator=rg,
    
               # define a numerical solver of adequate accuracy
               ode_solver='scipy.solve_ivp',
    
               # sets the input space to be field oriented voltage
               #control_space='dq', 
              )

env.reset()
tau = env._physical_system.tau
simulation_time = 20 # seconds
nb_steps = int(simulation_time // tau)

policy_kwargs = {
        'net_arch': [128,512,128]
        }

model = DQN(MlpPolicy, FlattenObservation(env), buffer_size = 100000, learning_starts=10000 , batch_size = 200, gamma = 0.5, policy_kwargs = policy_kwargs, verbose = 2).learn(total_timesteps=nb_steps)
    
obs = env.reset()    
obs = np.concatenate((obs[0], obs[1]))

for i in range(nb_steps):
    action, _states = model.predict(obs, deterministic=True)
    (state, reference), reward, done, _ = env.step(action)
    obs = np.concatenate((state, reference))
    env.render()

    if done:
        env.reset()