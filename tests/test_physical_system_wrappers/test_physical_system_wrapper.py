import pytest

import gym_electric_motor as gem
import tests.testing_utils as tu


class TestPhysicalSystemWrapper:

    @pytest.fixture
    def physical_system(self):
        return tu.DummyPhysicalSystem()

    @pytest.fixture
    def processor(self, physical_system):
        return gem.physical_system_wrappers.PhysicalSystemWrapper(physical_system=physical_system)

    @pytest.fixture
    def reset_processor(self, processor):
        processor.reset()
        return processor

    @pytest.fixture
    def double_wrapped(self, processor):
        return gem.physical_system_wrappers.PhysicalSystemWrapper(physical_system=processor)

    def test_action_space(self, processor, physical_system):
        assert processor.action_space == physical_system.action_space

    def test_state_space(self, processor, physical_system):
        assert processor.state_space == physical_system.state_space

    def test_physical_system(self, processor, physical_system):
        assert processor.physical_system == physical_system

    def test_unwrapped(self, double_wrapped, physical_system):
        assert double_wrapped.unwrapped == physical_system

    def test_nominal_state(self, processor, physical_system):
        assert all(processor.nominal_state == physical_system.nominal_state)

    def test_limits(self, processor, physical_system):
        assert all(processor.limits == physical_system.limits)

    def test_reset(self, processor, physical_system):
        assert all(processor.reset() == physical_system.state)

    @pytest.mark.parametrize(['action'], [[1]])
    def test_simulate(self, reset_processor, physical_system, action):
        state = reset_processor.simulate(action)
        assert state == physical_system.state
        assert action == physical_system.action
