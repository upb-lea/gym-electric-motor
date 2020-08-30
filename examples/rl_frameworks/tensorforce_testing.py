import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gym_electric_motor as gem
from gym_electric_motor.constraint_monitor import ConstraintMonitor
from gym_electric_motor.reference_generators import \
    MultipleReferenceGenerator,\
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad

from gym.spaces import Discrete, Box
from gym.wrappers import FlattenObservation, TimeLimit
from gym import ObservationWrapper

from tensorforce.environments import Environment
from tensorforce.agents import Agent
import json
from setting_environment import set_env

max_eps_steps = 10000
gem_env = set_env()
# creating tensorforce environment
tensor_env = Environment.create(environment=gem_env,
                                max_episode_timesteps=max_eps_steps)

"""
hyperparameter for agent_config:

    memory: size of replay-buffer
    batch_size: size of mini-batch used for training
    network: net-architect for dqn
    update_frequency: Frequency of updates
    start_updating: memory warm-up steps
    learning_rate for optimizer
    discount: gamma/ discount of future rewards
    target_sync_frequency: Target network gets updated 'sync_freq' steps
    target_update_weight: weight for target-network update

"""
epsilon_decay = {'type': 'decaying',
                 'decay': 'polynomial',
                 'decay_steps': 50000,
                 'unit': 'timesteps',
                 'initial_value': 1.0,
                 'decay_rate': 5e-2,
                 'final_value': 5e-2,
                 'power': 3.0}
net = [
    dict(type='dense', size=64, activation='relu'),
    dict(type='dense', size=64, activation='relu'),
    dict(type='linear', size=7)
]

agent_config = {
    'agent': 'dqn',
    'memory': 200000,
    'batch_size': 25,
    'network': net, #dict(type='auto', size=64, depth=2),
    'update_frequency': 1,
    'start_updating': 10000,
    'learning_rate': 1e-4,
    'discount': 0.99,
    'exploration': epsilon_decay,
    'target_sync_frequency': 1000,
    'target_update_weight': 1.0}

agent_path = 'saved_agents'
agent_name = 'dqn_64_64'

dqn_agent = Agent.load(
    directory=agent_path,
    filename=agent_name,
    format='pb-actonly',
    #environment=tensor_env,
    #**agent_config
)

# test agent
tau = 1e-5
steps = 5000#1000000

rewards = []
# lens = []
# obs_hist = []

states = []
references = []

obs = gem_env.reset()
#obs_hist.append(obs)
terminal = False
cum_rew = 0
step_counter = 0
eps_rew = 0

for step in tqdm(range(steps)):
    #while not terminal:
    gem_env.render()
    print(type(obs))
    actions = dqn_agent.act(obs, evaluation=True)
    obs, reward, terminal, _ = gem_env.step(action=actions)
    rewards.append(cum_rew)
    #obs_hist.append(obs)
    # dqn_agent.observe(terminal, reward=reward)
    cum_rew += reward
    eps_rew += reward

    if terminal:
        #lens.append(step_counter)
        step_counter = 0
        #print(f'Episode length: {episode_length} steps')
        obs = gem_env.reset()
        #obs_hist.append(obs)
        rewards.append(eps_rew)
        terminal = False
        eps_rew = 0
    step_counter += 1

print(f' \n Cumulated Reward per step is {cum_rew/steps} \n')
#print(f' \n Longest Episode: {np.amax(lens)} steps \n')



# test agent
# tau = 1e-5
# steps = 100000
#
# rewards = []
# lens = []
# obs_hist = []
#
# states = []
# references = []
#
# obs = gem_env_.reset()
# obs_hist.append(obs)
# terminal = False
# cum_rew = 0
# step_counter = 0
# eps_rew = 0
#
# for step in tqdm(range(steps)):
#     #while not terminal:
#     #gem_env_.render()
#     actions = dqn_agent.act(states=obs, independent=True)
#
#     obs, reward, terminal, _ = gem_env_.step(action=actions)
#     rewards.append(cum_rew)
#     obs_hist.append(obs)
#
#     # dqn_agent.observe(terminal, reward=reward)
#     cum_rew += reward
#     eps_rew += reward
#
#     if terminal:
#         lens.append(step_counter)
#         step_counter = 0
#         #print(f'Episode length: {episode_length} steps')
#         obs = gem_env_.reset()
#
#         obs_hist.append(obs)
#         rewards.append(eps_rew)
#
#         terminal = False
#         eps_rew = 0
#     step_counter += 1
#
#
# print(f' \n Cumulated Reward over {steps} steps is {cum_rew} \n')
# print(f' \