from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent.parent))
from stable_baselines3 import DQN
from setting_environment import set_env
import numpy as np
import matplotlib.pyplot as plt
"""
This example is based on stable baselines3 0.8.0a5. Since it is still being frequently updated 
some parts of this code might be broken or not the latest recommended way of working with it
"""


rewards = np.load(Path(__file__).parent / "saved_agents" / "PreTrainedEpisodeRewards.npy")[1:]
plt.grid(True)
plt.xlim(0,len(rewards))
plt.ylim(min(rewards), 0)
plt.yticks(np.arange(min(rewards), 1, 1.0))
plt.tick_params(axis = 'y', left = False, labelleft = False)
plt.xticks(np.arange(0, len(rewards), 10))
plt.xlabel('#Episode')
plt.ylabel('Mean Reward per Episode (Qualitativ)')
plt.plot(rewards)
plt.show()

N = 0
M = 0
gamma = 0.99
time_limit = False

env = set_env(time_limit = time_limit, gamma = gamma, N = N, M = M, training = False)
model = DQN.load(Path(__file__).parent / "saved_agents" / "TutorialPreTrainedAgent")  
obs = env.reset()   

visualization_steps = int(9e4) # currently this crashes for larger values
cum_rew_episode = 0
last_i = 0

print(f"Starting test visualization for {visualization_steps} steps\n")
for i in range(visualization_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    cum_rew_episode += reward
    env.render()
    if done:
        last_i = i
        cum_rew_episode = 0
        obs = env.reset() 
        
test_steps = int(1e6) #1 milion for stability reasons
print(f"\nStart Evaluation for {test_steps} steps, this may take some while")
for j in range(2):
    episode_lengths = []
    episode_step = 0
    cum_rew_testing_period = 0
    for i in range(test_steps):
        print(f"{i+1}", end = '\r')
        episode_step += 1
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        cum_rew_testing_period += reward
        if done:
            episode_lengths.append(episode_step)
            episode_step = 0
            obs = env.reset()
    print(f"Reward per step for testing period {j+1} with {test_steps} steps: {cum_rew_testing_period/test_steps:.4f} ")
    print(f"The average Episode length was: {round(np.mean(episode_lengths))} ")
