import pytest
import gym_electric_motor as gem
import numpy as np

control_tasks = ['TC', 'SC', 'CC']
action_types = ['Cont', 'Finite']
motors = ['SeriesDc', 'PermExDc', 'ExtExDc', 'ShuntDc', 'PMSM', 'EESM', 'SynRM', 'DFIM', 'SCIM']
versions = ['v0']
seeds = [123, 456, 789]


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('dc_motor', motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize('action_type', action_types)
@pytest.mark.parametrize('seed', seeds)
def test_seeding_same_env(dc_motor, control_task, action_type, version, no_of_steps, seed):
    """This test assures that an environment that is seeded two times with the same seed generates the same episodes."""
    env_id = f'{action_type}-{control_task}-{dc_motor}-{version}'
    env = gem.make(env_id)
    # Sample actions that are used in both executions
    actions = [env.action_space.sample() for _ in range(no_of_steps)]
    terminated = True
    states1 = []
    references1 = []
    rewards1 = []
    terminated1 = []
    # Execute the env
    for i in range(no_of_steps):
        if terminated:
            state, reference = env.reset(seed)
        (state, reference), reward, terminated, truncated, info = env.step(actions[i])
        rewards1.append(reward)
        states1.append(state)
        references1.append(reference)
        terminated1.append(terminated)

    terminated = True
    states2 = []
    references2 = []
    rewards2 = []
    terminated2 = []
    # Execute the environment again
    for i in range(no_of_steps):
        if terminated:
            state, reference = env.reset(seed)
        (state, reference), reward, terminated, truncated, info = env.step(actions[i])
        rewards2.append(reward)
        states2.append(state)
        references2.append(reference)
        terminated2.append(terminated)

    # Assure that the epsiodes of the initially and reseeded environment are equal
    references1 = np.array(references1).flatten()
    references2 = np.array(references2).flatten()
    assert(np.all(np.array(states1) == np.array(states2)))
    assert(np.all(np.array(references1).flatten() == np.array(references2).flatten()))
    assert(np.all(np.array(rewards1) == np.array(rewards2)))
    assert (np.all(np.array(terminated1) == np.array(terminated2)))


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('dc_motor', motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize('action_type', action_types)
@pytest.mark.parametrize('seed', seeds)
def test_seeding_new_env(dc_motor, control_task, action_type, version, no_of_steps, seed):
    """This test assures that two equal environments that are seeded with the same seed generate the same episodes."""
    env_id = f'{action_type}-{control_task}-{dc_motor}-{version}'
    env = gem.make(env_id)
    actions = [env.action_space.sample() for _ in range(no_of_steps)]
    terminated = True

    states1 = []
    references1 = []
    rewards1 = []
    terminated1 = []

    for i in range(no_of_steps):
        if terminated:
            state, reference = env.reset(seed)
        (state, reference), reward, terminated, truncated, info = env.step(actions[i])
        rewards1.append(reward)
        states1.append(state)
        references1.append(reference)
        terminated1.append(terminated)

    env = gem.make(env_id)
    terminated = True
    states2 = []
    references2 = []
    rewards2 = []
    terminated2 = []

    for i in range(no_of_steps):
        if terminated:
            state, reference = env.reset(seed)
        action = env.action_space.sample()
        assert action in env.action_space
        (state, reference), reward, terminated, truncated, info = env.step(actions[i])
        rewards2.append(reward)
        states2.append(state)
        references2.append(reference)
        terminated2.append(terminated)

    # Assure that the episodes of both environments are equal
    assert (np.all(np.array(states1) == np.array(states2)))
    assert (np.all(np.array(references1).flatten() == np.array(references2).flatten()))
    assert (np.all(np.array(rewards1) == np.array(rewards2)))
    assert (np.all(np.array(terminated1) == np.array(terminated2)))
