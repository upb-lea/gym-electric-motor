import sys
sys.path.insert(0, 'D:\GitHub\gym-electric-motor')

import gym_electric_motor as gem

if __name__ == '__main__':
    env = gem.make("Finite-CC-PMSM-v0")#,num_envs = 2)  # instantiate a discretely controlled PMSM
    env.reset()
    for _ in range(10000):
        (state, reference), reward, terminated, truncated, _ =  env.step(env.action_space.sample())  # pick random control actions
        if terminated:
            (states, references), _ = env.reset()
    env.close()