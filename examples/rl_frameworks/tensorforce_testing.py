import numpy as np
from tqdm import tqdm
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import tensorflow as tf
from setting_environment import set_env

max_eps_steps = 10000
gem_env = set_env(time_limit=False)
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
    dict(type='dense', size=64, activation='none'),
    #dict(type='linear', size=7)
]

agent_config = {
    'agent': 'dqn',
    'memory': 200000,
    'batch_size': 25,
    'network': net,
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
    environment=tensor_env,
    **agent_config
)
print('\n agent loaded \n')

# test agent
tau = 1e-5
steps = 1000000
rewards = []
states = []
references = []

obs = gem_env.reset()
terminal = False
cum_rew = 0
step_counter = 0
eps_rew = 0

for step in tqdm(range(steps)):
    # gem_env.render()
    actions = dqn_agent.act(obs, evaluation=True)
    obs, reward, terminal, _ = gem_env.step(action=actions)
    rewards.append(cum_rew)
    cum_rew += reward
    eps_rew += reward

    if terminal:
        step_counter = 0
        obs = gem_env.reset()
        rewards.append(eps_rew)
        terminal = False
        eps_rew = 0
    step_counter += 1

print(f' \n Cumulated Reward per step is {cum_rew/steps} \n')
