from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LeakyReLU
from keras.optimizers import Adamax, Adam
from rl.agents.dqn import DQNAgent
from rl.policy import MaxBoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gym.wrappers import FlattenObservation
import gym_electric_motor as gem
from gym_electric_motor.visualization import MotorDashboard
from gym_electric_motor.reference_generators import WienerProcessReferenceGenerator

if __name__ == '__main__':
    # Run the option parsing
    # Get the environment and extract the number of actions.
    env = gem.make(
        'emotor-dc-series-disc-v1',
        state_filter=['i'],
        # Pass an instance
        visualization=MotorDashboard(visu_period=0.5, plotted_variables=['omega', 'i', 'u']),
        converter='Disc-4QC',
        # Take standard class and pass parameters (Load)
        a=0, b=.1, c=1.1, j_load=0.4,
        # Pass a string (with extra parameters)
        ode_solver='euler', solver_kwargs={},
        # Pass a Class with extra parameters
        reference_generator=WienerProcessReferenceGenerator(reference_state='i', sigma_range=(5e-3, 5e-1))
    )
    nb_actions = env.action_space.n
    env = FlattenObservation(env)
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(4))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(4))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    memory = SequentialMemory(limit=15000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(eps=0.5), 'eps', 0.5, 0.01, 0, 20000)
    dqn = DQNAgent(
        model=model, policy=policy, nb_actions=nb_actions, memory=memory, gamma=0.5, batch_size=128,
        train_interval=1, memory_interval=1
    )

    dqn.compile(Adam(), metrics=['mse'])
    dqn.fit(env, nb_steps=200000, action_repetition=5, verbose=1, visualize=False, nb_max_episode_steps=50000)
    input()
    dqn.test(env, nb_episodes=3, nb_max_episode_steps=50000, visualize=True)
