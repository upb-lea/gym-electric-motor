import gym
import pytest
import numpy as np
import gym_electric_motor as gem

from .test_physical_system_wrapper import TestPhysicalSystemWrapper
from tests.testing_utils import DummyPhysicalSystem


class TestDeadTimeProcessor(TestPhysicalSystemWrapper):

    @pytest.fixture
    def processor(self, physical_system):
        return gem.physical_system_wrappers.DeadTimeProcessor(physical_system=physical_system)

    @pytest.fixture
    def unset_processor(self, physical_system):
        return gem.physical_system_wrappers.DeadTimeProcessor()

    @pytest.mark.parametrize('action', [np.array([5.0]), np.array([2.0])])
    def test_simulate(self, reset_processor, physical_system, action):
        state = reset_processor.simulate(action)
        assert state == physical_system.state
        assert all(physical_system.action == np.array([0.0]))

    @pytest.mark.parametrize('unset_processor', [
        gem.physical_system_wrappers.DeadTimeProcessor(steps=2),
        gem.physical_system_wrappers.DeadTimeProcessor(steps=1),
        gem.physical_system_wrappers.DeadTimeProcessor(steps=5),
    ])
    @pytest.mark.parametrize(
        ['action_space', 'actions', 'reset_action'],
        [
         [
            gym.spaces.Box(-100, 100, shape=(3,)),
            [np.array([1., 2., 3.]), np.array([0., 1., 2.]), np.array([-1, 2, 3])],
            np.array([0., 0., 0.])
         ],
         [
            gym.spaces.Box(-100, 100, shape=(1,)),
            [np.array([-1.]), np.array([0.]), np.array([-5.]), np.array([-6.]), np.array([-7.])],
            np.array([0.])
         ],
         [
            gym.spaces.MultiDiscrete([10, 20, 30]),
            [[5, 8, 7], [3, 4, 5], [0, 0, 1], [0, 1, 1], [0, 0, 1]],
            [0, 0, 0]
         ],
         [
             gym.spaces.Discrete(12),
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             0
         ]
        ]
    )
    def test_execution(self, unset_processor, physical_system, action_space, actions, reset_action):
        expected_actions = [reset_action] * unset_processor.dead_time + actions
        physical_system._action_space = action_space
        unset_processor.set_physical_system(physical_system)
        unset_processor.reset()
        for i, action in enumerate(actions):
            unset_processor.simulate(action)
            try:
                assert physical_system.action == expected_actions[i]
            except ValueError:
                assert all(physical_system.action == expected_actions[i])

    @pytest.mark.parametrize('processor', [gem.physical_system_wrappers.DeadTimeProcessor()])
    def test_false_action_space(self, processor, physical_system):
        physical_system._action_space = gym.spaces.MultiBinary(5)
        with pytest.raises(AssertionError):
            assert processor.set_physical_system(physical_system)

    @pytest.mark.parametrize('steps', [0, -10])
    def test_false_steps(self, steps):
        with pytest.raises(AssertionError):
            assert gem.physical_system_wrappers.DeadTimeProcessor(steps)
