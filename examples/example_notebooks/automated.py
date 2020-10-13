from pathlib import Path
import sys
sys.path.append(str(Path().resolve().parent.parent))
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
import time
from setting_environment import set_env


#Feature Engineering params
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

#env = set_env(time_limit, gamma, training=True, callbacks = [RewardLogger()])
env = set_env(time_limit, gamma, training=True)#, callbacks = [RewardLogger()])
nb_steps = int(simulation_time // tau)


start_time = time.time()

model = DQN(MlpPolicy, env, buffer_size=buffer_size, learning_starts=learning_starts ,train_freq=train_freq, batch_size=batch_size, gamma=gamma,
            policy_kwargs=policy_kwargs, exploration_fraction=exploration_fraction, target_update_interval=target_update_interval,
            verbose=verbose).learn(total_timesteps=nb_steps)




env = set_env(time_limit = time_limit, gamma = gamma, training = False)
obs = env.reset()   

test_steps = int(1e6) #1 milion for stability reasons
print(f"\nStart Evaluation for {test_steps} steps, this may take some while")
for j in range(2):
    episode_lengths = []
    episode_step = 0
    cum_rew_testing_period = 0
    for i in range(test_steps):
        #print(f"{i+1}", end = '\r')
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
