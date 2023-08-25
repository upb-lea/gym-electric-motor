import gymnasium as gym
import numpy as np
import gym_electric_motor as gem


if __name__ == '__main__':
    envs = gym.make_vec("CartPole-v1", num_envs=3, render_mode="human")

    _ = envs.reset(seed=42)

    actions = np.array([1, 0, 1])

    for i in range(10000): 
        observations, rewards, termination, truncation, infos = envs.step(actions)

    observations