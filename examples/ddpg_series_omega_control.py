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
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard

if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()
    # Create the environment
    # Default DcSeries Motor Parameters are changed to have more dynamic system and to see faster learning results.
    env = gem.make(
        'emotor-dc-series-cont-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard, visu_period=1,
        motor_parameter=dict(r_a=2.5, r_e=4.5, l_a=9.7e-3, l_e_prime=9.2e-3, l_e=9.2e-3, j_rotor=0.001),
        # Take standard class and pass parameters (Load)
        load_parameter=dict(a=0, b=.0, c=0.01, j_load=.001),
        reward_weights={'omega': 1000},
        reward_power=0.5,
        observed_states=None,  # Constraint violation monitoring is disabled for presentation purpose
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp', solver_kwargs=dict(method='BDF'),
        # Pass an instance
        reference_generator=WienerProcessReferenceGenerator(reference_state='omega', sigma_range=(5e-3, 1e-2))
    )
    # Keras-rl DDPG-agent accepts flat observations only
    env = FlattenObservation(env)
    nb_actions = env.action_space.shape[0]

    #  CAUTION: Do not use layers that behave differently in training and
    #  testing
    #  (e.g. dropout, batch-normalization, etc..)
    #  Reason is a bug in TF2 where not the learning_phase_tensor is extractable
    #  in order to put as an input to keras models
    #  https://stackoverflow.com/questions/58987264/how-to-get-learning-phase-in-tensorflow-2-eager
    #  https://stackoverflow.com/questions/58279628/what-is-the-difference-between-tf-keras-and-tf-python-keras?noredirect=1&lq=1
    #  https://github.com/tensorflow/tensorflow/issues/34508
    window_length = 1
    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(nb_actions, activation='sigmoid'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(window_length,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=(action_input, observation_input), outputs=x)
    print(critic.summary())

    # Create a replay memory
    memory = SequentialMemory(
        limit=10000,
        window_length=window_length
    )

    # Create a random process for exploration during training
    random_process = OrnsteinUhlenbeckProcess(
        theta=0.5,
        mu=0.0,
        sigma=0.2
    )

    # Create the agent
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        random_process=random_process,
        nb_steps_warmup_actor=2048,
        nb_steps_warmup_critic=1024,
        target_model_update=1000,
        gamma=0.95,
        batch_size=128,
        memory_interval=1
    )
    agent.compile((Adam(lr=1e-6), Adam(lr=1e-4)), metrics=['mae'])

    # Start training for 7.5M simulation steps (1.5M training steps with actions repeated 5 times)

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
