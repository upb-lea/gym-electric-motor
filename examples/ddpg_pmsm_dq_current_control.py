from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, \
    Concatenate
from tensorflow.keras import initializers, regularizers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess
from gym.wrappers import FlattenObservation
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator, \
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.physical_systems import ConstantSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple


class AppendLastActionWrapper(Wrapper):
    """
    The following environment considers the dead time in the real-world motor control systems.
    The real-world system changes its state, while the agent calculates the next action based on a previoulsly measured
    observation. Therefore, for the agents it seems as if the applied action effects the state one step delayed.
    (with a dead time of one time-step)

    For complete observability of the system at each time-step we append the last played action of the agent to the
    observation, because this action will be the one that is active in the next step.
    """
    def __init__(self, environment):
        super().__init__(environment)
        self.observation_space = Tuple((Box(
            np.concatenate((environment.observation_space[0].low, environment.action_space.low)),
            np.concatenate((environment.observation_space[0].high, environment.action_space.high))
        ), environment.observation_space[1]))

    def step(self, action):
        (state, ref), rew, term, info = self.env.step(action)
        state = np.concatenate((state, action))
        return (state, ref), rew, term, info

    def reset(self, **kwargs):
        state, ref = self.env.reset()
        state = np.concatenate((state, np.zeros(self.env.action_space.shape)))
        return state, ref


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    window_length = 1

    # Changing i_q reference and constant 0 i_d reference.
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')
    d_generator = ConstReferenceGenerator('i_sd', 0)
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    # Change of the default motor parameters.
    motor_parameter = dict(
        r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06
    )
    limit_values = dict(
        i=160*1.41,
        omega=12000 * np.pi / 30,
        u=450
    )
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}

    # Create the environment
    env = gem.make(
        'emotor-pmsm-cont-v1',
        # Pass a class with extra parameters
        visualization=MotorDashboard, visu_period=1,
        load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),
        control_space='dq',
        # Pass a string (with extra parameters)
        ode_solver='scipy.solve_ivp', solver_kwargs={},
        # Pass an instance
        reference_generator=rg,
        plotted_variables=['i_sq', 'i_sd', 'u_sq', 'u_sd'],
        reward_weights={'i_sq': 1000, 'i_sd': 1000},
        reward_power=0.5,
        observed_states=['i_sq', 'i_sd'],
        dead_time=True,
        u_sup=400,
        motor_parameter=motor_parameter,
        limit_values=limit_values,
        nominal_values=nominal_values,
        state_filter=['i_sq', 'i_sd', 'epsilon']
    )

    # Due to the dead time in the system, the
    env = AppendLastActionWrapper(env)

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

    actor = Sequential()
    actor.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    actor.add(Dense(16, activation='relu'))
    actor.add(Dense(17, activation='relu'))
    actor.add(Dense(nb_actions, kernel_initializer=initializers.RandomNormal(stddev=1e-5), activation='tanh',
                    kernel_regularizer=regularizers.l2(1e-2)))
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
        limit=5000,
        window_length=window_length
    )

    # Create a random process for exploration during training
    random_process = OrnsteinUhlenbeckProcess(
        theta=0.5,
        mu=0.0,
        sigma=0.1,
        dt=env.physical_system.tau,
        sigma_min=0.05,
        n_steps_annealing=85000,
        size=2
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
        gamma=0.9,
        batch_size=128,
        memory_interval=2
    )
    agent.compile([Adam(lr=3e-5), Adam(lr=3e-3)])

    # Start training for 75000 simulation steps
    agent.fit(
        env,
        nb_steps=75000,
        nb_max_start_steps=0,
        nb_max_episode_steps=10000,
        visualize=True,
        action_repetition=1,
        verbose=2,
        log_interval=10000,
        callbacks=[],

    )
    # Test the agent
    hist = agent.test(
        env,
        nb_episodes=5,
        action_repetition=1,
        visualize=True,
        nb_max_episode_steps=10000
    )

