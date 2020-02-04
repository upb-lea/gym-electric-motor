"""Run this file from within the 'examples' folder:
>>> cd examples
>>> python ddpg_series_omega_control.py
"""
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import GaussianWhiteNoiseProcess
from gym.wrappers import FlattenObservation
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard

if __name__ == '__main__':
    # Create the environment
    env = gem.make(
        'emotor-dc-series-cont-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard, visu_period=1,

        # Take standard class and pass parameters (Load)
        load_parameter=dict(a=0, b=.0, c=0.0, j_load=.5),

        # Pass a string (with extra parameters)
        ode_solver='euler', solver_kwargs={},
        # Pass an instance
        reference_generator=WienerProcessReferenceGenerator(reference_state='i')
    )
    # Keras-rl DDPG-agent accepts flat observations only
    env = FlattenObservation(env)
    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(16))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('linear'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)
    print(critic.summary())

    # Create a replay memory
    memory = SequentialMemory(
        limit=150000,
        window_length=1
    )

    # Create a random process for exploration during training
    random_process = GaussianWhiteNoiseProcess(
        mu=0.0,
        sigma=0.8,
        sigma_min=0.05,
        n_steps_annealing=650000
    )

    # Create the agent
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,
        nb_steps_warmup_actor=32,
        nb_steps_warmup_critic=32,
        target_model_update=1e-4,
        gamma=0.9,
        batch_size=32
    )
    agent.compile(Adam(lr=1e-4), metrics=['mae'])

    # Start training for 7.5M simulation steps (1.5M training steps with actions repeated 5 times)
    agent.fit(
        env,
        nb_steps=1500000,
        visualize=True,
        action_repetition=5,
        verbose=1,
        nb_max_start_steps=0,
        log_interval=10000,
        callbacks=[]
    )

    # Test the agent
    hist = agent.test(
        env,
        nb_episodes=10,
        action_repetition=1,
        visualize=True
    )
