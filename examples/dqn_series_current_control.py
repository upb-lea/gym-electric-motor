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
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import gym_electric_motor as gem
from gym_electric_motor.reward_functions import WeightedSumOfErrors
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator


if __name__ == '__main__':

    # Default DcSeries Motor Parameters are changed to have more dynamic system and to see faster learning results.
    env = gem.make(
        'emotor-dc-series-disc-v1',
        state_filter=['omega', 'i'],
        # Pass an instance
        reward_function=WeightedSumOfErrors(observed_states='i'),
        visualization=MotorDashboard(update_period=1e-2, visu_period=1e-1, plotted_variables=['omega', 'i', 'u']),
        converter='Disc-1QC',
        # Take standard class and pass parameters (Load)
        motor_parameter=dict(r_a=15e-3, r_e=15e-3, l_a=1e-3, l_e=1e-3),
        load_parameter=dict(a=0, b=.1, c=.1, j_load=0.04),
        # Pass a string (with extra parameters)
        ode_solver='euler', solver_kwargs={},
        # Pass a Class with extra parameters
        reference_generator=WienerProcessReferenceGenerator(reference_state='i', sigma_range=(3e-3, 3e-2))
    )
    env = FlattenObservation(env)

    nb_actions = env.action_space.n
    window_length = 1

    model = Sequential()
    model.add(Flatten(input_shape=(window_length,) + env.observation_space.shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=15000, window_length=window_length)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.2), 'eps', 0.2, 0.01, 0, 20000)
    dqn = DQNAgent(
        model=model, policy=policy, nb_actions=nb_actions, memory=memory, gamma=0.9, batch_size=128,
        train_interval=1, memory_interval=1
    )

    dqn.compile(Adam(lr=1e-4), metrics=['mse'])
    dqn.fit(env, nb_steps=200000, action_repetition=1, verbose=2, visualize=True, nb_max_episode_steps=50000,
            log_interval=10000)
    dqn.test(env, nb_episodes=3, nb_max_episode_steps=50000, visualize=True)
