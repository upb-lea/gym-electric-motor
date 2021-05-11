"""Run this file from within the 'examples' folder:
>> cd examples
>> python dqn_series_current_control.py
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym.wrappers import FlattenObservation
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator

'''
This example shows how we can use GEM to train a reinforcement learning agent to control the motor current
of a DC series motor. In this scenario, the state space is continuous while the action space is discrete.
We use a deep Q learning agent to determine which action must be taken on a finite-control-set
'''

if __name__ == '__main__':

    # Define the drive environment
    # Default DcSeries Motor Parameters are changed to have more dynamic system and to see faster learning result
    env = gem.make(
        # Define the series DC motor with finite-control-set
        'Finite-CC-SeriesDc-v0',

        # Defines the utilized power converter, which determines the action space
        # 'Disc-1QC' is our notation for a discontinuous one-quadrant converter,
        # which is a one-phase buck converter with available actions 'switch on' and 'switch off'
        converter='Finite-1QC',

        # Define which states will be shown in the state observation (what we can "measure")
        state_filter=['omega', 'i'],
    )

    # Now, the environment will output states and references separately
    state, ref = env.reset()

    # For data processing we sometimes want to flatten the env output,
    # which means that the env will only output one array that contains states and references consecutively
    env = FlattenObservation(env)
    obs = env.reset()

    # Read the number of possible actions for the given env
    # this allows us to define a proper learning agent for this task
    nb_actions = env.action_space.n

    window_length = 1

    # Define an artificial neural network to be used within the agent
    model = Sequential()
    # The network's input fits the observation space of the env
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    # The network output fits the action space of the env
    model.add(Dense(nb_actions, activation='linear'))

    # Define a memory buffer for the agent, allows to learn from past experiences
    memory = SequentialMemory(limit=15000, window_length=window_length)

    # Define the policy which the agent will use for training and testing
    # in this case, we use an epsilon greedy policy with decreasing epsilon for training
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.2),
                                  attr='eps',
                                  value_max=0.2,   # initial value of epsilon (during training)
                                  value_min=0.01,  # final value of epsilon (during training)
                                  value_test=0,    # epsilon during testing, epsilon=0 => deterministic behavior
                                  nb_steps=20000   # annealing interval
                                                   # (duration for the transition from value_max to value_min)
                                  )

    # Create the agent for deep Q learning
    dqn = DQNAgent(
        # Pass the previously defined characteristics
        model=model,
        policy=policy,
        nb_actions=nb_actions,
        memory=memory,

        # Define the overall training parameters
        gamma=0.9,
        batch_size=128,
        train_interval=1,
        memory_interval=1
    )

    # Compile the model within the agent (making it ready for training)
    # using ADAM optimizer
    dqn.compile(Adam(lr=1e-4), metrics=['mse'])

    # Start training the agent
    dqn.fit(env,
            nb_steps=200000,             # number of training steps
            action_repetition=1,
            verbose=2,
            visualize=True,              # use the environment's visualization (the dashboard)
            nb_max_episode_steps=50000,  # maximum length of one episode
                                         # (episodes end prematurely when drive limits are violated)
            log_interval=10000)

    # Test the agent (without exploration noise, as we set value_test=0 within our policy)
    dqn.test(env,
             nb_episodes=3,
             nb_max_episode_steps=50000,
             visualize=True)
