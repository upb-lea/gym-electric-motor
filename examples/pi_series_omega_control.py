import gym
import numpy as np
from examples.agents.simple_controllers import Controller
import gym_electric_motor

# Initialize the environment
env_name = 'emotor-dc-series-cont-v0'

motor_type = env_name.split('-')[2]


env = gym.make(
    env_name,
    episode_length=10000,
    on_dashboard=['True'],
    reward_weight=[["omega", 1.0]],
    reward_fct='swsae',
    limit_observer='no_punish',
    safety_margin=1.3
)
agent = 'cascaded_PI_controller'  # Specify the controller


# necessary to fit the controller parameters to the plant and specify the controller
params = env.get_motor_param()
params['env_name'] = env_name
controller = Controller(agent, params)

controller.action_space = env.action_space
controller.observation_space = env.observation_space


# Reset before every episode starts
omega, torque, i_, u_in, *ref = env.reset()

# state = [omega, armature current, excitation current]
# armature and excitation current are the same at the series DC motor
state = np.array([omega, i_, i_])

done = False
k = 0
cum_rew = 0
while not done:
    # Visualization call
    env.render()

    # The controller determines the next action
    action = controller.control(state, ref[0], i_)

    # The action is applied to the environment which returns the next observation
    (omega, torque, i_, u_in, u_sup, *ref), rew, done, _ = env.step(action)

    # next motor state
    state = np.array([omega, i_, i_])

    # cumulative reward
    cum_rew += rew
    k += 1
print('Episode Reward: ', cum_rew)


