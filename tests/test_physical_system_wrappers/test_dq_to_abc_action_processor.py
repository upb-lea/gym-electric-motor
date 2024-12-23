import gymnasium
import pytest
import numpy as np
import gym_electric_motor as gem

from ..testing_utils import DummyPhysicalSystem
from .test_physical_system_wrapper import TestPhysicalSystemWrapper


class TestDqToAbcActionProcessor(TestPhysicalSystemWrapper):
    @pytest.fixture
    def physical_system(self):
        ps = DummyPhysicalSystem(state_names=["omega", "epsilon", "i"])
        ps.electrical_motor = gem.physical_systems.PermanentMagnetSynchronousMotor()
        return ps

    @pytest.fixture
    def processor(self, physical_system):
        return gem.physical_system_wrappers.DqToAbcActionProcessor.make(
            "PMSM", physical_system=physical_system
        )

    def test_action_space(self, processor, physical_system):
        space = gymnasium.spaces.Box(-1, 1, shape=(2,), dtype=np.float64)
        assert processor.action_space == space

    @pytest.mark.parametrize(
        ["dq_action", "state", "abc_action"],
        [
            (
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
            ),
            (
                np.array([0.0, 1.0]),
                np.array([12.8, 0.123, 0.0]),
                np.array([-0.45595422,  0.99874252, -0.54278829]),
            ),
            (
                np.array([0.0, 0.5]),
                np.array([-10.0, 0.123, 0.0]),
                np.array([ 0.36929056, -0.47656712,  0.10727657]),
            ),
        ],
    )
    def test_simulate(
        self, reset_processor, physical_system, dq_action, state, abc_action
    ):
        reset_processor._state = state
        reset_processor.simulate(dq_action)

        assert all(np.isclose(reset_processor.action, abc_action))
