import pytest
import gym_electric_motor as gem
import numpy as np

control_tasks = ['TC', 'SC', 'CC']
action_types = ['AbcCont', 'DqCont', 'Finite']
ac_motors = ['PMSM', 'SynRM', 'SCIM', 'DFIM']
versions = ['v0']
seeds = [123, 456, 789]


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('ac_motor', ac_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize('action_type', action_types)
@pytest.mark.parametrize('seed', seeds)
def test_seeding_same_env(ac_motor, control_task, action_type, version, no_of_steps, seed):
    """This test assures that a reseeding of the same environment leads to the same episodes again."""
    env_id = f'{action_type}-{control_task}-{ac_motor}-{version}'
    env = gem.make(env_id)
    # Initial seed
    env.seed(seed)
    # Sample some actions
    actions = [env.action_space.sample() for _ in range(no_of_steps)]
    done = True
    # Data of the initial seed
    states1 = []
    references1 = []
    rewards1 = []
    done1 = []
    # Simulate the environment with the sampled actions
    for i in range(no_of_steps):
        if done:
            state, reference = env.reset()
        (state, reference), reward, done, info = env.step(actions[i])
        rewards1.append(reward)
        states1.append(state)
        references1.append(reference)
        done1.append(done)

    # Reseed the environment
    env.seed(seed)
    done = True
    states2 = []
    references2 = []
    rewards2 = []
    done2 = []

    # Execute the reseeded env with the same actions
    for i in range(no_of_steps):
        if done:
            state, reference = env.reset()
        (state, reference), reward, done, info = env.step(actions[i])
        rewards2.append(reward)
        states2.append(state)
        references2.append(reference)
        done2.append(done)

    # Assure that the data of the initial and second seeding of the environment are equal
    references1 = np.array(references1).flatten()
    references2 = np.array(references2).flatten()
    assert(np.all(np.array(states1) == np.array(states2)))
    assert(np.all(np.array(references1).flatten() == np.array(references2).flatten()))
    assert(np.all(np.array(rewards1) == np.array(rewards2)))
    assert (np.all(np.array(done1) == np.array(done2)))


@pytest.mark.parametrize('no_of_steps', [100])
@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('ac_motor', ac_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize('action_type', action_types)
@pytest.mark.parametrize('seed', seeds)
def test_seeding_new_env(ac_motor, control_task, action_type, version, no_of_steps, seed):
    """This test assures that two equal environments that are seeded with the same seed generate the same episodes."""
    env_id = f'{action_type}-{control_task}-{ac_motor}-{version}'
    env = gem.make(env_id)
    # Seed the first environment
    env.seed(seed)
    # Sample actions
    actions = [env.action_space.sample() for _ in range(no_of_steps)]
    done = True
    # Episode data of the first environment
    states1 = []
    references1 = []
    rewards1 = []
    done1 = []
    # Simulate the environment
    for i in range(no_of_steps):
        if done:
            state, reference = env.reset()
        (state, reference), reward, done, info = env.step(actions[i])
        rewards1.append(reward)
        states1.append(state)
        references1.append(reference)
        done1.append(done)
    # Create a new environment with the same id
    env = gem.make(env_id)
    # Seed the environment with the same seed
    env.seed(seed)
    done = True
    states2 = []
    references2 = []
    rewards2 = []
    done2 = []
    # Execute the new environment
    for i in range(no_of_steps):
        if done:
            state, reference = env.reset()
        action = env.action_space.sample()
        assert action in env.action_space
        (state, reference), reward, done, info = env.step(actions[i])
        rewards2.append(reward)
        states2.append(state)
        references2.append(reference)
        done2.append(done)

    # Assure that the episodes of both environments are equal
    assert (np.all(np.array(states1) == np.array(states2)))
    assert (np.all(np.array(references1).flatten() == np.array(references2).flatten()))
    assert (np.all(np.array(rewards1) == np.array(rewards2)))
    assert (np.all(np.array(done1) == np.array(done2)))
