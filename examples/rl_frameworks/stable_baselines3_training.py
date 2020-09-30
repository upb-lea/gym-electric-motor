from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent.parent))
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import time
from setting_environment import set_env
from gym_electric_motor.core import Callback
import numpy as np

"""
This example is based on stable baselines3 0.8.0a5. Since it is still being frequently updated 
some parts of this code might be broken or not the latest recommended way of working with it
"""

class RewardLogger(Callback):
    def __init__(self):
        self._step_rewards = []
        self._mean_episode_rewards = []

    def on_step_end(self):
        self._step_rewards.append(self._env._reward)
    
    def on_reset_begin(self):
        self._mean_episode_rewards.append(np.mean(self._step_rewards))
        self._step_rewards = []
        
    def on_close(self):
        np.save(Path(__file__).parent / "saved_agents" / "PreTrainedEpisodeRewards.npy", np.array(self._mean_episode_rewards))

#Feature Engineering params
N = 0 # N last actions will be appended to the observation
M = 0 # M last actions will be appended to the observation
time_limit = True # Whether the environment terminates its episodes after 10000 steps

#Training parameters. 
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

env = set_env(time_limit, gamma, N, M, training=True, callbacks = [RewardLogger()])

nb_steps = int(simulation_time // tau)


start_time = time.time()

model = DQN(MlpPolicy, env, buffer_size=buffer_size, learning_starts=learning_starts ,train_freq=train_freq, batch_size=batch_size, gamma=gamma,
            policy_kwargs=policy_kwargs, exploration_fraction=exploration_fraction, target_update_interval=target_update_interval,
            verbose=verbose).learn(total_timesteps=nb_steps)

print(f'Execution time of stable baselines3 DQN is: {time.time()-start_time:.2f} seconds')

#in case your want to save the model for further evalutation
model.save(Path(__file__).parent / "saved_agents" / "TutorialPreTrainedAgent")
env.close()