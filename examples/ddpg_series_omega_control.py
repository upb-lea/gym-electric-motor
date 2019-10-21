import gym
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import gym_electric_motor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess, AnnealedGaussianProcess
from rl.callbacks import FileLogger


# Create the environment
env_name = 'emotor-dc-series-cont-v0'
env = gym.make(
    env_name,
    episode_length=10000,
    on_dashboard=['omega', 'u'],
    reward_weight=[["omega", 1.0]],
    reward_fct='swsae',
    limit_observer='no_punish',
    safety_margin=1.3
)

nb_actions = env.action_space.shape[0]

action_input = Input(shape=(nb_actions,), name='action_input')

# Actor Model
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))

# Critic Model
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)

# Create a replay memory
memory = SequentialMemory(
    limit=150000,
    window_length=1,
    ignore_episode_boundaries=True
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
    visualize=False,
    action_repetition=5,
    verbose=2,
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