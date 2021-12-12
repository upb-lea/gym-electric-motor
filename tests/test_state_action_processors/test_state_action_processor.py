import pytest
import gym_electric_motor as gem
from ..testing_utils import DummyPhysicalSystem


class TestStateActionProcessor:

    @pytest.fixture
    def physical_system(self):
        return DummyPhysicalSystem()

    @pytest.fixture
    def processor(self, physical_system):
        return gem.state_action_processors.StateActionProcessor(physical_system=physical_system)

    @pytest.fixture
    def double_wrapped(self, processor):
        return gem.state_action_processors.StateActionProcessor(physical_system=processor)

    def test_action_space(self, processor, physical_system):
        assert processor.action_space == physical_system.action_space

    def test_state_space(self, processor, physical_system):
        assert processor.state_space == physical_system.state_space

    def test_physical_system(self, processor, physical_system):
        assert processor.physical_system == physical_system

    def test_unwrapped(self, double_wrapped, physical_system):
        assert double_wrapped.unwrapped == physical_system

    def test_nominal_state(self, processor, physical_system):
        assert processor.nominal_state == physical_system.nominal_state

    def test_limits(self, processor, physical_system):
        assert processor.limits == physical_system.limits

    def test_reset(self, processor, physical_system):
        assert processor.reset() == physical_system.state

    @pytest.mark.parametrize(['action'], [[1]])
    def test_simulate(self, processor, physical_system, action):
        state = processor.simulate(action)
        assert state == physical_system.state
        assert action == physical_system.action
