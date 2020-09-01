from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import time
from setting_environment import set_env

max_eps_steps = 10000
simulation_steps = 500000
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

# creating agent via dictionary
dqn_agent = Agent.create(agent=agent_config, environment=tensor_env)
# create agent trainer
runner = Runner(agent=dqn_agent, environment=tensor_env)
# time training
start_time = time.time()
runner.run(num_timesteps=simulation_steps)
end_time = time.time()

agent_path = 'saved_agents'
agent_name = 'dqn_64_64'
runner.agent.save(directory=agent_path, filename=agent_name)
runner.close()

print('\n agent saved \n')
print(f'\n Execution time of tensorforce dqn-training is:'
      f' 'f'{end_time-start_time:.2f} seconds \n ')
