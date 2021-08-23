import pytest


import gym_electric_motor as gem


control_tasks = ['TC', 'SC', 'CC']
ac_action_types = ['AbcCont', 'Finite', 'DqCont']
dc_action_types = ['Cont', 'Finite']
ac_motors = ['PMSM', 'SynRM', 'SCIM', 'DFIM']
dc_motors = ['SeriesDc', 'ShuntDc', 'PermExDc', 'ExtExDc']
versions = ['v0']


@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('dc_motor', dc_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize(['action_type', 'tau'], zip(dc_action_types, [1e-4, 1e-5]))
def test_tau_dc(dc_motor, control_task, action_type, version, tau):
    env_id = f'{action_type}-{control_task}-{dc_motor}-{version}'
    env = gem.make(env_id)
    assert env.physical_system.tau == tau


@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('ac_motor', ac_motors)
@pytest.mark.parametrize('control_task', control_tasks)
@pytest.mark.parametrize(['action_type', 'tau'], zip(ac_action_types, [1e-4, 1e-5, 1e-4]))
def test_tau_ac(ac_motor, control_task, action_type, version, tau):
    env_id = f'{action_type}-{control_task}-{ac_motor}-{version}'
    env = gem.make(env_id)
    assert env.physical_system.tau == tau


@pytest.mark.parametrize('version', versions)
@pytest.mark.parametrize('ac_motor', ac_motors)
@pytest.mark.parametrize(
    ['control_task', 'referenced_states'], zip(control_tasks, [['torque'], ['omega'], ['i_sd', 'i_sq']])
)
@pytest.mark.parametrize('action_type', ac_action_types)
def test_referenced_states_ac(ac_motor, control_task, action_type, version, referenced_states):
    env_id = f'{action_type}-{control_task}-{ac_motor}-{version}'
    env = gem.make(env_id)
    assert env.reference_generator.reference_names == referenced_states
