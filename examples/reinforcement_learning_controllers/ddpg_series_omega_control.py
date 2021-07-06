"""Run this file from within the 'examples' folder:
>> cd examples
>> python ddpg_series_omega_control.py
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, \
     Concatenate
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from gym.wrappers import FlattenObservation
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard

'''
This example shows how we can use GEM to train a reinforcement learning agent to control the motor speed
of a DC series motor. The state and action space is continuous.
We use a deep-deterministic-policy-gradient (DDPG) agent to
determine which action must be taken on a continuous-control-set
'''

if __name__ == '__main__':

    # Define the drive environment
    env = gem.make(
        # Define the speed control series DC motor environment with continuous-control-set
        'Cont-SC-SeriesDc-v0',
    )

    # For data processing we want to flatten the env output,
    # which means that the env will only output one array that contains states and references consecutively
    state, ref = env.reset()
    env = FlattenObservation(env)
    obs = env.reset()

    # Read the dimension of the action space
    # this allows us to define a proper learning agent for this task
    nb_actions = env.action_space.shape[0]

    #  CAUTION: Do not use layers that behave differently in training and testing
    #  (e.g. dropout, batch-normalization, etc..)
    #  Reason is a bug in TF2 where not the learning_phase_tensor is extractable
    #  in order to put as an input to keras models
    #  https://stackoverflow.com/questions/58987264/how-to-get-learning-phase-in-tensorflow-2-eager
    #  https://stackoverflow.com/questions/58279628/what-is-the-difference-between-tf-keras-and-tf-python-keras?noredirect=1&lq=1
    #  https://github.com/tensorflow/tensorflow/issues/34508

    # Define how many past observations we want the control agent to process each step
    # for this case, we assume to pass only the single most recent observation
    window_length = 1

    # Define an artificial neural network to be used within the agent as actor
    # (using keras sequential)
    actor = Sequential()
    # The network's input fits the observation space of the env
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    # The network output fits the action space of the env
    actor.add(Dense(nb_actions, activation='sigmoid'))
    print(actor.summary())

    # Define another artificial neural network to be used within the agent as critic
    # note that this network has two inputs
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
    # (using keras functional API)
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)
    print(critic.summary())

    # Define a memory buffer for the agent, allows to learn from past experiences
    memory = SequentialMemory(
        limit=10000,
        window_length=window_length
    )

    # Create a random process for exploration during training
    # this is essential for the DDPG algorithm
    random_process = OrnsteinUhlenbeckProcess(
        theta=0.5,
        mu=0.0,
        sigma=0.2
    )

    # Create the agent for DDPG learning
    agent = DDPGAgent(
        # Pass the previously defined characteristics
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,

        # Define the overall training parameters
        nb_steps_warmup_actor=2048,
        nb_steps_warmup_critic=1024,
        target_model_update=1000,
        gamma=0.95,
        batch_size=128,
        memory_interval=1
    )

    # Compile the function approximators within the agent (making them ready for training)
    # Note that the DDPG agent uses two function approximators, hence we define two optimizers here
    agent.compile((Adam(lr=1e-6), Adam(lr=1e-4)), metrics=['mae'])

    # Start training for 1.5 million simulation steps
    agent.fit(
        env,
        nb_steps=1500000,
        visualize=True,
        action_repetition=1,
        verbose=1,
        nb_max_start_steps=0,
        nb_max_episode_steps=10000,
        log_interval=10000,
        callbacks=[]
    )

    # Test the agent
    hist = agent.test(
        env,
        nb_episodes=10,
        action_repetition=1,
        nb_max_episode_steps=5000,
        visualize=True
    )
