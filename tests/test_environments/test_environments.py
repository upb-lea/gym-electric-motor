import pytest


import gym_electric_motor as gem


control_tasks = ['TC', 'SC', 'CC']
action_types = ['Cont', 'Finite']
ac_motors = ['PMSM', 'SynRM', 'SCIM', 'DFIM']
dc_motors = ['SeriesDc', 'ShuntDc', 'PermExDc', 'ExtExDc']
versions = ['v0']


@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('motor', ac_motors + dc_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize(['action_type', 'tau'], zip(action_types, [1e-4, 1e-5, 1e-4]))
def test_tau(motor, control_task, action_type, version, tau):
    env_id = f'{action_type}-{control_task}-{motor}-{version}'
    env = gem.make(env_id)
    assert env.physical_system.tau == tau


@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('ac_motor', ac_motors)
@pytest.mark.parametrize(
    ['control_task', 'referenced_states'], zip(control_tasks, [['torque'], ['omega'], ['i_sd', 'i_sq']])
)
@pytest.mark.parametrize('action_type', action_types)
def test_referenced_states_ac(ac_motor, control_task, action_type, version, referenced_states):
    env_id = f'{action_type}-{control_task}-{ac_motor}-{version}'
    env = gem.make(env_id)
    assert env.reference_generator.reference_names == referenced_states
