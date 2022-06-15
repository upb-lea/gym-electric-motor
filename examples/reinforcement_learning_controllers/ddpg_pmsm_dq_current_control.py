from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, \
    Concatenate
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from gym.wrappers import FlattenObservation
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reference_generators import MultipleReferenceGenerator, ConstReferenceGenerator, \
    WienerProcessReferenceGenerator
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.visualization.motor_dashboard_plots import MeanEpisodeRewardPlot
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym.core import Wrapper
from gym.spaces import Box, Tuple
from gym_electric_motor.constraints import SquaredConstraint
from gym_electric_motor.physical_system_wrappers import DqToAbcActionProcessor, DeadTimeProcessor

'''
This example shows how we can use GEM to train a reinforcement learning agent to control the current within
a permanent magnet synchronous motor three-phase drive.
It is assumed that we have direct access to signals within the flux-oriented dq coordinate system.
Hence, we assume to directly control the current in the dq frame.
The state and action space is continuous.
We use a deep-deterministic-policy-gradient (DDPG) agent
to determine which action must be taken on a continuous-control-set
'''


class AppendLastActionWrapper(Wrapper):
    """
    The following environment considers the dead time in the real-world motor control systems.
    The real-world system changes its state, while the agent simultaneously calculates the next action based on a
    previously measured observation.
    Therefore, for the agents it seems as if the applied action affects the environment with one step delay
    (with a dead time of one time step).

    As a measure of feature engineering we append the last selected action to the observation of each time step,
    because this action will be the one that is active while the agent has to make the next decision.
    """
    def __init__(self, environment):
        super().__init__(environment)
        # append the action space dimensions to the observation space dimensions
        self.observation_space = Tuple((Box(
            np.concatenate((environment.observation_space[0].low, environment.action_space.low)),
            np.concatenate((environment.observation_space[0].high, environment.action_space.high))
        ), environment.observation_space[1]))

    def step(self, action):

        (state, ref), rew, term, info = self.env.step(action)

        # extend the output state by the selected action
        state = np.concatenate((state, action))

        return (state, ref), rew, term, info

    def reset(self, **kwargs):

        state, ref = self.env.reset()

        # extend the output state by zeros after reset
        # no action can be appended yet, but the dimension must fit
        state = np.concatenate((state, np.zeros(self.env.action_space.shape)))

        return state, ref


if __name__ == '__main__':

    # Define reference generators for both currents of the flux oriented dq frame
    # d current reference is chosen to be constantly at zero to simplify this showcase scenario
    d_generator = ConstReferenceGenerator('i_sd', 0)
    # q current changes dynamically
    q_generator = WienerProcessReferenceGenerator(reference_state='i_sq')

    # The MultipleReferenceGenerator allows to apply these references simultaneously
    rg = MultipleReferenceGenerator([d_generator, q_generator])

    # Set the electric parameters of the motor
    motor_parameter = dict(
        r_s=15e-3, l_d=0.37e-3, l_q=1.2e-3, psi_p=65.6e-3, p=3, j_rotor=0.06
    )

    # Change the motor operational limits (important when limit violations can terminate and reset the environment)
    limit_values = dict(
        i=160*1.41,
        omega=12000 * np.pi / 30,
        u=450
    )

    # Change the motor nominal values
    nominal_values = {key: 0.7 * limit for key, limit in limit_values.items()}
    physical_system_wrappers = (
        DeadTimeProcessor(),
        DqToAbcActionProcessor.make('PMSM'),
    )
    
    # Create the environment
    env = gem.make(
        # Choose the permanent magnet synchronous motor with continuous-control-set
        'Cont-CC-PMSM-v0',
        # Pass a class with extra parameters
        physical_system_wrappers=physical_system_wrappers,
        visualization=MotorDashboard(
            state_plots=['i_sq', 'i_sd'],
            action_plots='all',
            reward_plot=True,
            additional_plots=[MeanEpisodeRewardPlot()]
        ),
        # Set the mechanical load to have constant speed
        load=ConstantSpeedLoad(omega_fixed=1000 * np.pi / 30),

        # Define which numerical solver is to be used for the simulation
        ode_solver='scipy.ode',

        # Pass the previously defined reference generator
        reference_generator=rg,

        reward_function=dict(
            # Set weighting of different addends of the reward function
            reward_weights={'i_sq': 1000, 'i_sd': 1000},
            # Exponent of the reward function
            # Here we use a square root function
            reward_power=0.5,
        ),

        # Define which state variables are to be monitored concerning limit violations
        # Here, only overcurrent will lead to termination
        constraints=(SquaredConstraint(('i_sq', 'i_sd')),),

        # Consider converter dead time within the simulation
        # This means that a given action will show effect only with one step delay
        # This is realistic behavior of drive applications
        
        # Set the DC-link supply voltage
        supply=dict(
            u_nominal=400
        ),

        motor=dict(
            # Pass the previously defined motor parameters
            motor_parameter=motor_parameter,

            # Pass the updated motor limits and nominal values
            limit_values=limit_values,
            nominal_values=nominal_values,
        ),
        # Define which states will be shown in the state observation (what we can "measure")
        state_filter=['i_sd', 'i_sq', 'epsilon'],
    )

    # Now we apply the wrapper defined at the beginning of this script
    env = AppendLastActionWrapper(env)

    # We flatten the observation (append the reference vector to the state vector such that
    # the environment will output just a single vector with both information)
    # This is necessary for compatibility with kerasRL2
    env = FlattenObservation(env)

    # Read the dimension of the action space
    # this allows us to define a proper learning agent for this task
    nb_actions = env.action_space.shape[0]

    #  CAUTION: Do not use layers that behave differently in training and
    #  testing
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
    actor.add(Dense(17, activation='relu'))
    # The network output fits the action space of the env
    actor.add(Dense(
        nb_actions,
        kernel_initializer=initializers.RandomNormal(stddev=1e-5),
        activation='tanh',
        kernel_regularizer=regularizers.l2(1e-2))
    )
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
        limit=5000,
        window_length=window_length
    )

    # Create a random process for exploration during training
    # this is essential for the DDPG algorithm
    random_process = OrnsteinUhlenbeckProcess(
        theta=0.5,
        mu=0.0,
        sigma=0.1,
        dt=env.physical_system.tau,
        sigma_min=0.05,
        n_steps_annealing=85000,
        size=2
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
        gamma=0.9,
        batch_size=128,
        memory_interval=2
    )

    # Compile the function approximators within the agent (making them ready for training)
    # Note that the DDPG agent uses two function approximators, hence we define two optimizers here
    agent.compile([Adam(lr=3e-5), Adam(lr=3e-3)])

    # Start training for 75 k simulation steps
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
