import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper

"""
This example is based on stable baselines3 0.8.0a5. Since it is still being frequently updated 
some parts of this code might be broken or not the latest recommended way of working with it
"""

gamma = 0.99
tau = 1e-5
simulation_time = 5 # seconds
buffer_size = 200000 #number of old obersation steps saved
learning_starts = 10000 # memory warmup
train_freq = 1 # prediction network gets an update each train_freq's step
batch_size = 25 # mini batch size drawn at each update step
policy_kwargs = {
        'net_arch': [64,64] # hidden layer size of MLP
        }

exploration_fraction = 0.1 # Fraction of training steps the epsilon decays 
target_update_interval = 1000 # Target network gets updated each target_update_interval's step
verbose = 1 # verbosity of stable basline's prints




class EpsilonWrapper(ObservationWrapper):
    """Changes Epsilon in a flattened observation to cos(epsilon) and sin(epsilon)"""
    def __init__(self,env, epsilon_idx):
        super(EpsilonWrapper, self).__init__(env)
        self.EPSILON_IDX = epsilon_idx
        new_low = np.concatenate((self.env.observation_space.low[:self.EPSILON_IDX], np.array([-1.]), self.env.observation_space.low[self.EPSILON_IDX:]))
        new_high = np.concatenate((self.env.observation_space.high[:self.EPSILON_IDX], np.array([1.]), self.env.observation_space.high[self.EPSILON_IDX:]))

        self.observation_space = Box(new_low, new_high)
        
    def observation(self,observation):
        cos_eps = np.cos(observation[self.EPSILON_IDX]*np.pi)
        sin_eps = np.sin(observation[self.EPSILON_IDX]*np.pi)
        observation = np.concatenate((observation[:self.EPSILON_IDX], np.array([cos_eps, sin_eps]), observation[self.EPSILON_IDX+1:]))
        return observation
        

class SqdCurrentMonitor:
    """
    monitor for squared currents:
    
    i_sd**2 + i_sq**2 < 1.5 * nominal_limit 
    """
    
    def __call__(self, state, observed_states, k, physical_system):
        self.I_SD_IDX = physical_system.state_names.index('i_sd')
        self.I_SQ_IDX = physical_system.state_names.index('i_sq')
        # normalize to limit_values, since state is normalized
        nominal_values = physical_system.nominal_state / abs(physical_system.limits)
        limits = 1.5 * nominal_values
        # calculating squared currents as observed measure 
        sqd_currents = state[self.I_SD_IDX]**2 + state[self.I_SQ_IDX]**2
        
        return (sqd_currents > limits[self.I_SD_IDX] or sqd_currents > limits[self.I_SQ_IDX])
        

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

env = gem.make(# define a PMSM with discrete action space
               "PMSMDisc-v1",
               
               #visualize the results
               visualization = MotorDashboard(plots = ['i_sq', 'i_sd', 'reward']),
    
               #parameterize the PMSM
               motor_parameter=motor_parameter,
               #converter = converter,
    
               #update the limitations of the state space
               limit_values=limit_values,
               nominal_values=nominal_values,
               
               # define the DC link voltage
               u_sup=u_sup, 
               
               # define the speed at which the motor is operated - should be drawn randomly at each episode
               load='ConstSpeedLoad',
               load_initializer = {'states': {'omega': 1000 * np.pi / 30,},
                                   'interval': [[-4000*2*np.pi/60, 4000*2*np.pi/60]],
                                   'random_init': 'uniform',
                   },
               #gem.physical_systems.ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30), 
               
               # define the duration of one sampling step
               tau=tau,
               
               # turn off terminations via limit violation and parameterize the reward function
               reward_function=gem.reward_functions.WeightedSumOfErrors(observed_states=['i_sq', 'i_sd'], 
                                                                        reward_weights={'i_sq': 1, 'i_sd': 1},
                                                                        constraint_monitor = SqdCurrentMonitor(),
                                                                        gamma = gamma,
                                                                        reward_power=1
                                                                       ),
               # define the reference generator
               reference_generator=rg,
    
               # define a numerical solver of adequate accuracy
               ode_solver='euler',
                  )
       

eps_idx = env._physical_system.state_names.index('epsilon')
ml = env._physical_system._mechanical_load
env =  TimeLimit(EpsilonWrapper(FlattenObservation(env), eps_idx), 10000)
env.reset()

#Since action 0 == action 7 the action space may be districted to not "confuse" the neural network
env.action_space = Discrete(7)

nb_steps = int(simulation_time // tau)



model = DQN(MlpPolicy, env, buffer_size=buffer_size, learning_starts=learning_starts ,train_freq=train_freq, batch_size=batch_size, gamma=gamma,
            policy_kwargs=policy_kwargs, exploration_fraction=exploration_fraction, target_update_interval=target_update_interval,
            verbose=verbose).learn(total_timesteps=nb_steps)

#in case your want to save the model for further evalutation
model.save("dqn_PMSM")

    
obs = env.reset()    
cum_rew = 0
last_i = 0
for i in range(nb_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    cum_rew += reward
    env.render()
    if done:
        print(f'Episode had {i-last_i} steps')
        print(f'The reward per step was {cum_rew/(i-last_i)}')
        last_i = i
        cum_rew = 0
        env.reset()