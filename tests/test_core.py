import pytest
import numpy as np
from tests.testing_utils import DummyPhysicalSystem, DummyReferenceGenerator, DummyRewardFunction, DummyVisualization,\
    DummyCallback, DummyConstraintMonitor, DummyConstraint, mock_instantiate, instantiate_dict
from gym.spaces import Tuple, Box
import gym_electric_motor
from gym_electric_motor.core import ElectricMotorEnvironment, RewardFunction, \
    ReferenceGenerator, PhysicalSystem, ConstraintMonitor, Constraint
from gym_electric_motor.constraints import LimitConstraint
import gym
import gym_electric_motor as gem


class TestElectricMotorEnvironment:
    test_class = ElectricMotorEnvironment
    key = ''

    @pytest.fixture
    def env(self):
        ps = DummyPhysicalSystem()
        rg = DummyReferenceGenerator()
        rf = DummyRewardFunction()
        vs = ()
        cb = DummyCallback()
        cm = DummyConstraintMonitor(2)
        env = self.test_class(
            physical_system=ps,
            reference_generator=rg,
            reward_function=rf,
            visualizations=vs,
            constraints=cm,
            callbacks=[cb]
        )
        return env

    def test_make(self):
        if self.key != '':
            env = gem.make(self.key)
            assert type(env) == self.test_class

    @pytest.mark.parametrize(
        "physical_system, reference_generator, reward_function, state_filter, visualization, callbacks",
        [
            (
                DummyPhysicalSystem(), DummyReferenceGenerator(), DummyRewardFunction(), None, (), []
            ),
            (
                DummyPhysicalSystem(2), DummyReferenceGenerator(), DummyRewardFunction(), ['dummy_state_0'],
                (), [DummyCallback()]
            ),
            (
                DummyPhysicalSystem(10),
                DummyReferenceGenerator(),
                DummyRewardFunction(observed_states=['dummy_state_0']), ['dummy_state_0', 'dummy_state_2'],
                (),
                [DummyCallback(), DummyCallback()],
            ),
        ]
    )
    def test_initialization(
            self, monkeypatch, physical_system, reference_generator, reward_function, state_filter, visualization,
            callbacks
    ):

        env = gym_electric_motor.core.ElectricMotorEnvironment(
            physical_system=physical_system,
            reference_generator=reference_generator,
            reward_function=reward_function,
            visualizations=visualization,
            state_filter=state_filter,
            callbacks=callbacks,
        )

        # Assertions that the Keys are passed correctly to the instantiate fct
        assert physical_system == env.physical_system
        assert reference_generator == env.reference_generator
        assert reward_function == env.reward_function

        if state_filter is None:
            assert Tuple((
                    physical_system.state_space,
                    reference_generator.reference_space
            )) == env.observation_space, 'Wrong observation space'
        else:
            state_idxs = np.isin(physical_system.state_names, state_filter)
            state_space = Box(
                physical_system.state_space.low[state_idxs],
                physical_system.state_space.high[state_idxs],
            )
            assert Tuple(
                (state_space, reference_generator.reference_space)
            ) == env.observation_space, 'Wrong observation space'
        assert env.reward_range == reward_function.reward_range, 'Wrong reward range'
        for callback in callbacks:
            assert callback._env == env

    def test_reset(self, env):
        ps = env.physical_system
        rg = env.reference_generator
        rf = env.reward_function
        cbs = env._callbacks
        rf.last_state = rf.last_reference = ps.state = rg.get_reference_state = None
        # Initial amount of resets
        for callback in cbs:
            assert callback.reset_begin == 0
            assert callback.reset_end == 0
        state, ref = env.reset()
        # The corresponding callback functions should've been called
        for callback in cbs:
            assert callback.reset_begin == 1
            assert callback.reset_end == 1
        assert (state, ref) in env.observation_space, 'Returned values not in observation space'
        assert np.all(np.all(state == ps.state)), 'Returned state is not the physical systems state'
        assert np.all(ref == rg.reference_observation), 'Returned reference is not the reference generators reference'
        assert np.all(state == rg.get_reference_state), 'Incorrect state passed to the reference generator'
        assert rf.last_state == state, 'Incorrect state passed to the Reward Function'
        assert rf.last_reference == rg.reference_array, 'Incorrect Reference passed to the reward function'

    @pytest.mark.parametrize('action, set_done', [(0, False), (-1, False), (1, False), (2, True)])
    def test_step(self, env, action, set_done):
        ps = env.physical_system
        rg = env.reference_generator
        rf = env.reward_function
        cbs = env._callbacks
        cm = env.constraint_monitor
        cm.constraints[0].violation_degree = float(set_done)
        with pytest.raises(Exception):
            env.step(action), 'Environment goes through the step without previous reset'
        env.reset()
        # Callback's step initial step values
        for callback in cbs:
            assert callback.step_begin == 0
            assert callback.step_end == 0
        (state, reference), reward, done, _ = env.step(action)
        # Each of callback's step functions were called in step
        for callback in cbs:
            assert callback.step_begin == 1
            assert callback.step_end == 1
        assert np.all(state == ps.state[env.state_filter]), 'Returned state and Physical Systems state are not equal'
        assert rg.get_reference_state == ps.state,\
            'State passed to the Reference Generator not equal to Physical System state'
        assert rg.get_reference_obs_state == ps.state, \
            'State passed to the Reference Generator not equal to Physical System state'
        assert ps.action == action, 'Action passed to Physical System not equal to selected action'
        assert reward == -1 if set_done else 1
        assert done == set_done
        # If episode terminated, no further step without reset
        if set_done:
            with pytest.raises(Exception):
                env.step(action)

    def test_close(self, env):
        ps = env.physical_system
        rg = env.reference_generator
        rf = env.reward_function
        cbs = env._callbacks
        # Callback's step initial close value
        for callback in cbs:
            assert callback.close == 0
        env.close()
        # Callback's close function was called on close
        for callback in cbs:
            assert callback.close == 1
        assert ps.closed, 'Physical System was not closed'
        assert rf.closed, 'Reward Function was not closed'
        assert rg.closed, 'Reference Generator was not closed'

    @pytest.mark.parametrize("reference_generator", (DummyReferenceGenerator(),))
    def test_reference_generator_change(self, env, reference_generator):
        env.reset()
        env.reference_generator = reference_generator
        assert env.reference_generator == reference_generator, 'Reference Generator was not changed'
        # Without Reset an Exception has to be thrown
        with pytest.raises(Exception):
            env.step(env.action_space.sample()), 'After Reference Generator change was no reset required'
        env.reset()
        # No Exception raised
        env.step(env.action_space.sample())

    @pytest.mark.parametrize("reward_function", (DummyRewardFunction(),))
    def test_reward_function_change(self, env, reward_function):
        env.reset()
        reward_function.set_modules(
            physical_system=env.physical_system, reference_generator=env.reference_generator,
            constraint_monitor=env.constraint_monitor
        )
        env.reward_function = reward_function
        assert env.reward_function == reward_function, 'Reward Function was not changed'
        # Without Reset an Exception has to be thrown
        with pytest.raises(Exception):
            env.step(env.action_space.sample()), 'After Reward Function change was no reset required'
        env.reset()
        # No Exception raised
        env.step(env.action_space.sample())

    @pytest.mark.parametrize(
        "number_states, state_filter, expected_result", (
            (1, ['dummy_state_0'], [10]),
            (3, ['dummy_state_0', 'dummy_state_1', 'dummy_state_2'], [10, 20, 30]),
            (3, ['dummy_state_1'], [20])
        )
    )
    def test_limits(self, number_states, state_filter, expected_result):
        ps = DummyPhysicalSystem(state_length=number_states)
        rg = DummyReferenceGenerator()
        rf = DummyRewardFunction()
        vs = DummyVisualization()
        cm = DummyConstraintMonitor(1)
        env = self.test_class(
            physical_system=ps,
            reference_generator=rg,
            reward_function=rf,
            visualization=vs,
            state_filter=state_filter,
            constraints=cm
        )
        assert all(env.limits == expected_result)


class TestReferenceGenerator:
    test_object = None
    initial_state = np.array([1, 2, 3, 4, 5]) / 5
    _reference_value = np.array([0.5])
    _observation = np.zeros(5)
    counter_obs = 0

    @pytest.fixture
    def reference_generator(self, monkeypatch):
        monkeypatch.setattr(ReferenceGenerator, "get_reference_observation", self.mock_get_reference_observation)
        monkeypatch.setattr(ReferenceGenerator, "get_reference", self.mock_get_reference)
        rg = ReferenceGenerator()
        rg._referenced_states = np.array([True, False])
        return rg

    def mock_get_reference_observation(self, initial_state):
        assert all(initial_state == self.initial_state)
        self.counter_obs += 1
        return self._observation

    def mock_get_reference(self, initial_state):
        assert all(initial_state == self.initial_state)
        return self._reference_value

    def test_reset(self, reference_generator):
        reference, observation, kwargs = reference_generator.reset(self.initial_state)
        assert all(reference == reference_generator.get_reference(self.initial_state))
        assert all(observation == reference_generator.get_reference_observation(self.initial_state))
        assert kwargs is None

    def test_referenced_states(self, reference_generator):
        assert reference_generator.referenced_states.dtype == bool


class TestPhysicalSystem:

    def test_initialization(self):
        action_space = gym.spaces.Discrete(3)
        state_space = gym.spaces.Box(-1, 1, shape=(3,))
        state_names = [f'dummy_state_{i}' for i in range(3)]
        tau = 1
        ps = PhysicalSystem(action_space, state_space, state_names, tau)
        assert ps.action_space == action_space
        assert ps.state_space == state_space
        assert ps.state_names == state_names
        assert ps.tau == tau
        assert ps.k == 0


class TestConstraintMonitor:

    @pytest.mark.parametrize(
        ['ps', 'limit_constraints', 'expected_observed_states'], [
            [DummyPhysicalSystem(3), ['dummy_state_0', 'dummy_state_2'], ['dummy_state_0', 'dummy_state_2']],
            [DummyPhysicalSystem(1), ['dummy_state_0'], ['dummy_state_0']],
            [DummyPhysicalSystem(2), ['all_states'], ['dummy_state_0', 'dummy_state_1']]
        ]
                             )
    def test_limit_constraint_setting(self, ps, limit_constraints, expected_observed_states):
        cm = ConstraintMonitor(limit_constraints=limit_constraints)
        cm.set_modules(ps)
        assert cm.constraints[0]._observed_state_names == expected_observed_states

    @pytest.mark.parametrize('constraints', [
        [lambda state: 0.0, DummyConstraint(), DummyConstraint()]
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(1)])
    def test_additional_constraint_setting(self, ps, constraints):
        cm = ConstraintMonitor(additional_constraints=constraints)
        cm.set_modules(ps)
        assert all(constraint in cm.constraints for constraint in constraints)

    @pytest.mark.parametrize('additional_constraints', [
        [lambda state: 0.0, DummyConstraint(), DummyConstraint()]
    ])
    @pytest.mark.parametrize('limit_constraints', [
        ['all_states'], ['dummy_state_0'], []
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(1)])
    def test_set_modules(self, ps, limit_constraints, additional_constraints):
        cm = ConstraintMonitor(limit_constraints, additional_constraints)
        cm.set_modules(ps)
        assert all(
            constraint.modules_set for constraint in cm.constraints if isinstance(constraint, DummyConstraint)
        )
        assert all(
            constraint._observed_states is not None for constraint in cm.constraints
            if isinstance(constraint, LimitConstraint)
        )

    @pytest.mark.parametrize(['violations', 'expected_violation_degree'], [
        [(0.5, 0.8, 0.0, 1.0), 1.0],
        [(0.5, 0.8, 0.0), 0.8],
        [(0.5,), 0.5],
        [(0.0,), 0.0],
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(1)])
    def test_max_merge_violations(self, ps, violations, expected_violation_degree):
        cm = ConstraintMonitor(merge_violations='max')
        cm.set_modules(ps)
        cm._merge_violations(violations)

    @pytest.mark.parametrize(['violations', 'expected_violation_degree'], [
        [(0.5, 0.8, 0.0, 1.0), 1.0],
        [(0.5, 0.8, 0.0), 0.9],
        [(0.5,), 0.5],
        [(0.0,), 0.0],
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(1)])
    def test_product_merge_violations(self, ps, violations, expected_violation_degree):
        cm = ConstraintMonitor(merge_violations='product')
        cm.set_modules(ps)
        cm._merge_violations(violations)

    @pytest.mark.parametrize(['merging_fct', 'violations', 'expected_violation_degree'], [
        [lambda *violations: 1.0, (0.5, 0.8, 0.0, 1.0), 1.0],
        [lambda *violations: 0.756, (0.5, 0.8, 0.0), 0.756],
        [lambda *violations: 0.123, (0.5,), 0.123],
        [lambda *violations: 0.0, (0.0,), 0.0],
    ])
    @pytest.mark.parametrize('ps', [DummyPhysicalSystem(1)])
    def test_callable_merge_violations(self, ps, merging_fct, violations, expected_violation_degree):
        cm = ConstraintMonitor(merge_violations=merging_fct)
        cm.set_modules(ps)
        cm._merge_violations(violations)

    @pytest.mark.parametrize(['violations', 'expected_violation_degree'], [
        [(0.5, 0.8, 0.0, 1.0), 1.0],
        [(0.5, 0.8, 0.0), 0.756],
        [(0.5,), 0.123],
        [(0.0,), 0.0],
    ])
    @pytest.mark.parametrize(['ps', 'state'], [[DummyPhysicalSystem(1), np.array([1.0])]])
    def test_check_constraints(self, ps, state, violations, expected_violation_degree):
        passed_violations = []

        def merge_violations(*violation_degrees):
            passed_violations.append(violation_degrees)
            return expected_violation_degree

        constraints = [DummyConstraint(viol_degree) for viol_degree in violations]
        cm = ConstraintMonitor(additional_constraints=constraints, merge_violations=merge_violations)
        cm.set_modules(ps)
        degree = cm.check_constraints(state)
        assert degree == expected_violation_degree
        assert all(passed == expected for passed, expected in zip(passed_violations[0][0], violations))
