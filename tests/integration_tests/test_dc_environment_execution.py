import pytest
import gym_electric_motor as gem
import numpy as np

control_tasks = ['TC', 'SC', 'CC']
action_types = ['Cont', 'Finite']
dc_motors = ['SeriesDc', 'PermExDc', 'ExtExDc', 'ShuntDc']
versions = ['v0']


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('dc_motor', dc_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize('action_type', action_types)
def test_execution(dc_motor, control_task, action_type, version, no_of_steps):
    env_id = f'{action_type}-{control_task}-{dc_motor}-{version}'
    env = gem.make(env_id)
    done = True
    for i in range(no_of_steps):
        if done:
            observation = env.reset()
            state, reference = observation
        action = env.action_space.sample()
        assert action in env.action_space
        observation, reward, done, info = env.step(action)
        assert not np.any(np.isnan(observation[0])), 'An invalid nan-value is in the state.'
        assert not np.any(np.isnan(observation[1])), 'An invalid nan-value is in the reference.'
        assert info == {}
        assert type(reward) in [float, np.float64, np.float32], 'The Reward is not a scalar floating point value.'
        assert not np.isnan(reward), 'Invalid nan-value as reward.'
        # Only the shape is monitored here. The states and references may lay slightly outside of the specified space.
        # This happens if limits are violated or if some states are not observed to lay within their limits.
        assert observation[0].shape == env.observation_space[0].shape, 'The shape of the state is incorrect.'
        assert observation[1].shape == env.observation_space[1].shape, 'The shape of the reference is incorrect.'

