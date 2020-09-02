from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent.parent))
from stable_baselines3 import DQN
from setting_environment import set_env

"""
This example is based on stable baselines3 0.8.0a5. Since it is still being frequently updated 
some parts of this code might be broken or not the latest recommended way of working with it
"""

gamma = 0.99

env = set_env(time_limit = False, gamma = gamma)
model = DQN.load(Path(__file__).parent / "saved_agents" / "sb3_dqn_PMSM")  
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
        print(f'Episode had {i-last_i} steps')
        print(f'The reward per step was {cum_rew_episode/(i-last_i)}')
        last_i = i
        cum_rew_episode = 0
        env.reset() 
        
print("\nStart Evaluation, this may take some while")
test_steps = int(1e6) #1 milion for stability reasons
cum_rew_testing_period = 0
for j in range(3):
    for i in range(test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        cum_rew_testing_period += reward
        if done:
            env.reset()
    print(f"Reward per step for testing period {j+1} with {test_steps} steps: {cum_rew_testing_period/test_steps} ")
    cum_rew_testing_period = 0