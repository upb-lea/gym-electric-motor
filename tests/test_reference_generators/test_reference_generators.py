import gym
from numpy.random import seed
import numpy.random as rd
import pytest
import gym_electric_motor as gem
from tests.test_core import TestReferenceGenerator
from gym_electric_motor.reference_generators import ConstReferenceGenerator, MultipleReferenceGenerator
from gym_electric_motor.reference_generators.switched_reference_generator import SwitchedReferenceGenerator
from gym_electric_motor.reference_generators.sawtooth_reference_generator import SawtoothReferenceGenerator
from gym_electric_motor.reference_generators.sinusoidal_reference_generator import SinusoidalReferenceGenerator
from gym_electric_motor.reference_generators.step_reference_generator import StepReferenceGenerator
from gym_electric_motor.reference_generators.triangle_reference_generator import TriangularReferenceGenerator
from gym_electric_motor.reference_generators.wiener_process_reference_generator import WienerProcessReferenceGenerator
from gym_electric_motor.reference_generators.subepisoded_reference_generator import SubepisodedReferenceGenerator
from gym_electric_motor.reference_generators.zero_reference_generator import ZeroReferenceGenerator
import gym_electric_motor.reference_generators.switched_reference_generator as swrg
import gym_electric_motor.reference_generators.subepisoded_reference_generator as srg
import gym_electric_motor.reference_generators.sawtooth_reference_generator as sawrg
import gym_electric_motor.reference_generators.wiener_process_reference_generator as wrg
from gym_electric_motor.core import ReferenceGenerator
from ..testing_utils import *
import numpy as np


# region second version


class TestSwitchedReferenceGenerator:
    """
    class for testing the switched reference generator
    """
    _reference_generator = []
    _physical_system = None
    _sub_generator = []
    # pre defined test values and expected results
    _kwargs = {'test': 42}
    _reference = 0.8
    _reference_observation = np.array([0.5, 0.6, 0.8, 0.15])
    _trajectory = np.ones((4, 15))
    _initial_state = np.array([0.1, 0.25, 0.6, 0.78])
    _initial_reference = 0.8

    # counter
    _monkey_instantiate_counter = 0
    _monkey_super_set_modules_counter = 0
    _monkey_dummy_set_modules_counter = 0
    _monkey_reset_reference_counter = 0
    _monkey_dummy_reset_counter = 0

    @pytest.fixture(scope='function')
    def setup(self):
        """
        fixture to reset the counter and _reference_generator
        :return:
        """
        self._reference_generator = []

    def monkey_instantiate(self, superclass, instance, **kwargs):
        """
        mock function for utils.instantiate
        Function tests if the instance and superclass are as expected.
        A dummy reference generator is instantiated.
        :param superclass:
        :param instance:
        :param kwargs:
        :return: DummyReferenceGenerator
        """
        assert superclass == ReferenceGenerator, 'superclass is not ReferenceGenerator as expected'
        assert instance == self._sub_generator[self._monkey_instantiate_counter], \
            'Instance is not the expected reference generator'
        dummy = DummyReferenceGenerator()
        self._reference_generator.append(dummy)
        self._monkey_instantiate_counter += 1
        return dummy

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    def monkey_dummy_set_modules(self, physical_system):
        """
        mock function for set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_dummy_set_modules_counter += 1
        assert self._physical_system == physical_system, 'physical system is not the expected instance'

    def monkey_reset_reference(self):
        """
        mock function for reset_reference()
        :return:
        """
        self._monkey_reset_reference_counter += 1

    def monkey_dummy_reset(self, initial_state, initial_reference):
        """
        mock function for DummyReferenceGenerator.reset()
        :param initial_state:
        :param initial_reference:
        :return:
        """
        if type(initial_state == self._initial_state) is bool:
            assert initial_state == self._initial_state, 'passed initial state is not the expected one'
        else:
            assert all(initial_state == self._initial_state), 'passed initial state is not the expected one'
        assert initial_reference == self._initial_reference, 'passed initial reference is not the expected one'
        self._monkey_dummy_reset_counter += 1
        return self._reference, self._reference_observation, self._trajectory

    def monkey_dummy_get_reference(self, state, **kwargs):
        """
        mock function for DummyReferenceGenerator.get_reference()
        :param state:
        :param kwargs:
        :return:
        """
        assert all(state == self._initial_state), 'passed state is not the expected one'
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'
        return self._reference

    def monkey_dummy_get_reference_observation(self, state, **kwargs):
        """
        mock function for DummyReferenceGenerator.get_reference_observation()
        :param state:
        :param kwargs:
        :return:
        """
        assert all(state == self._initial_state), 'passed state is not the expected one'
        assert self._kwargs == kwargs, 'Different additional arguments. Keep in mind None and {}.'
        return self._reference_observation

    @pytest.mark.parametrize(
        "sub_generator",
        [
            [SinusoidalReferenceGenerator()],
            [WienerProcessReferenceGenerator()],
            [StepReferenceGenerator()],
            [TriangularReferenceGenerator()],
            [SawtoothReferenceGenerator()],
            [
                SinusoidalReferenceGenerator(), WienerProcessReferenceGenerator(), StepReferenceGenerator(),
                TriangularReferenceGenerator(), SawtoothReferenceGenerator()
            ],
            [SinusoidalReferenceGenerator(), WienerProcessReferenceGenerator()],
            [StepReferenceGenerator(), TriangularReferenceGenerator(), SawtoothReferenceGenerator()]
        ]
    )
    @pytest.mark.parametrize("p", [None, [0.1, 0.2, 0.3, 0.2, 0.1]])
    @pytest.mark.parametrize(
        "super_episode_length, expected_sel",
        [((200, 500), (200, 500)), (100, (100, 101)), (500, (500, 501))]
    )
    def test_init(self, monkeypatch, setup, sub_generator, p, super_episode_length, expected_sel):
        """
        test function for the initialization of a switched reference generator with different combinations of reference
        generators
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :param sub_generator: list of sub generators
        :param p: probabilities for the sub generators
        :param super_episode_length: range of teh episode length of the switched reference generator
        :param expected_sel: expected switched reference generator episode length
        :return:
        """
        # setup test scenario
        self._sub_generator = sub_generator
        # call function to test
        test_object = SwitchedReferenceGenerator(sub_generator, p=p, super_episode_length=super_episode_length)
        # verify the expected results
        assert len(test_object._sub_generators) == len(sub_generator), 'unexpected number of sub generators'
        assert test_object._current_episode_length == 0, 'The current episode length is not 0.'
        assert test_object._super_episode_length == expected_sel, 'super episode length is not as expected'
        assert test_object._current_ref_generator in sub_generator
        assert test_object._sub_generators == list(sub_generator), 'Other sub generators than expected'

    def test_set_modules(self, monkeypatch, setup):
        """
        test set_modules()
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :return:
        """
        # setup test scenario
        sub_generator = [
            SinusoidalReferenceGenerator(reference_state='dummy_state_0'),
            WienerProcessReferenceGenerator(reference_state='dummy_state_0')
        ]
        reference_states = [1, 0, 0, 0, 0, 0, 0]
        # Override reference spaces
        sub_generator[0]._limit_margin = (1, 0)
        sub_generator[1]._limit_margin = (0, 0.5)

        expected_space = gym.spaces.Box(-1, 0.5, shape=(1,))
        self._sub_generator = sub_generator
        test_object = SwitchedReferenceGenerator(sub_generator)

        self._physical_system = DummyPhysicalSystem(7)
        self._physical_system._state_space = gym.spaces.Box(-1, 1, shape=self._physical_system.state_space.shape)
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert all(test_object.reference_space.low == expected_space.low), 'Lower limit of the reference space is not 0'
        assert test_object.reference_space.high == expected_space.high, 'Upper limit of the reference space is not 1'
        assert np.all(test_object._referenced_states == reference_states), 'referenced states are not the expected ones'

    @pytest.mark.parametrize("initial_state", [None, [0.8, 0.6, 0.4, 0.7]])
    @pytest.mark.parametrize("initial_reference", [None, 0.42])
    def test_reset(self, monkeypatch, setup, initial_state, initial_reference):
        """
        test reset()
        :param monkeypatch:
        :param setup: fixture to reset the counters and _reference_generators
        :param initial_state: tested initial state
        :param initial_reference: tested initial reference
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(SwitchedReferenceGenerator, '_reset_reference', self.monkey_reset_reference)
        sub_generator = [SinusoidalReferenceGenerator(), WienerProcessReferenceGenerator()]
        self._sub_generator = sub_generator
        test_object = SwitchedReferenceGenerator(sub_generator)
        monkeypatch.setattr(test_object._sub_generators[0], 'reset', self.monkey_dummy_reset)
        self._initial_state = initial_state
        self._initial_reference = initial_reference
        # call function to test
        res_0, res_1, res_2 = test_object.reset(initial_state, initial_reference)
        # verify the expected results
        assert self._monkey_dummy_reset_counter == 1, 'reset of sub generators is not called once'
        assert res_0 == self._reference, 'reference is not the expected one'
        assert all(res_1 == self._reference_observation), 'observation is not the expected one '
        assert sum(sum(abs(res_2 - self._trajectory))) < 1E-6, \
            'absolute difference of reference trajectory to the expected is larger than 1e-6'

    def test_get_reference(self, monkeypatch):
        """
        test get_reference()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        sub_generator = [DummyReferenceGenerator(), DummyReferenceGenerator()]
        self._sub_generator = sub_generator
        test_object = SwitchedReferenceGenerator(sub_generator)
        monkeypatch.setattr(DummyReferenceGenerator, 'get_reference', self.monkey_dummy_get_reference)
        # call function to test
        reference = test_object.get_reference(self._initial_state, **self._kwargs)
        # verify the expected results
        assert reference == self._reference, 'reference is not the expected one'

    @pytest.mark.parametrize("k, current_episode_length, k_new, reset_counter", [(10, 15, 11, 0), (10, 10, 11, 1)])
    def test_get_reference_observation(self, monkeypatch, k, current_episode_length, k_new, reset_counter):
        """
        test get_reference_observation()
        :param monkeypatch:
        :param k: used time step
        :param current_episode_length: value of the current episode length
        :param k_new: expected next time step
        :param reset_counter: expected values for reset_counter
        :return:
        """
        # setup test scenario
        sub_generator = [SinusoidalReferenceGenerator(), WienerProcessReferenceGenerator()]
        self._sub_generator = sub_generator

        test_object = SwitchedReferenceGenerator(sub_generator)
        monkeypatch.setattr(test_object, '_k', k)
        monkeypatch.setattr(test_object, '_current_episode_length', current_episode_length)
        monkeypatch.setattr(test_object, '_reset_reference', self.monkey_reset_reference)
        monkeypatch.setattr(test_object, '_reference', self._initial_reference)
        monkeypatch.setattr(test_object._current_ref_generator, 'reset', self.monkey_dummy_reset)
        monkeypatch.setattr(test_object._current_ref_generator, 'get_reference_observation',
                            self.monkey_dummy_get_reference_observation)
        # call function to test
        observation = test_object.get_reference_observation(self._initial_state, **self._kwargs)
        # verify the expected results
        assert all(observation == self._reference_observation), 'observation is not the expected one'
        assert test_object._k == k_new, 'unexpected new step in the reference'
        assert self._monkey_reset_reference_counter == reset_counter, 'reset_reference called unexpected often'

    def test_reset_reference(self, monkeypatch):
        sub_reference_generators = [DummyReferenceGenerator(), DummyReferenceGenerator(), DummyReferenceGenerator()]
        probabilities = [0.2, 0.5, 0.3]
        super_episode_length = (1, 10)
        test_object = SwitchedReferenceGenerator(
            sub_generators=sub_reference_generators, super_episode_length=super_episode_length, p=probabilities
        )
        test_object.seed(np.random.SeedSequence(123))
        test_object._reset_reference()
        assert test_object._k == 0
        assert test_object._super_episode_length[0] \
               <= test_object._current_episode_length\
               <= test_object._super_episode_length[1]
        assert test_object._current_ref_generator in sub_reference_generators


class TestWienerProcessReferenceGenerator:
    """
    class for testing the wiener process reference generator
    """
    _kwargs = None
    _current_value = None

    # counter
    _monkey_get_current_value_counter = 0

    def monkey_get_current_value(self, value):
        if self._current_value is not None:
            assert value == self._current_value
        self._monkey_get_current_value_counter += 1
        return value

    @pytest.mark.parametrize('sigma_range', [(2e-3, 2e-1)])
    def test_init(self, monkeypatch, sigma_range):
        """
        test init()
        :param monkeypatch:
        :param sigma_range: used range of sigma
        :return:
        """
        # setup test scenario
        # call function to test
        test_object = WienerProcessReferenceGenerator(sigma_range=sigma_range)
        # verify the expected results

        assert test_object._sigma_range == sigma_range, 'sigma range is not passed correctly'

    def test_reset_reference(self, monkeypatch):
        sigma_range = 1e-2
        episode_length = 10
        limit_margin = (-1, 1)
        reference_value = 0.5
        expected_reference = np.array([0.6, 0.4, 1, 1, 0.5, 0.2, -1, -0.9, -1, -0.6])

        class MonkeyRandomGenerator:
            _rg = DummyRandom(exp_loc=0, exp_scale=sigma_range, exp_size=episode_length)

            def normal(self, loc=0, scale=1, size=1):
                return self._rg.monkey_random_normal(loc, scale, size)

        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        test_object = WienerProcessReferenceGenerator(sigma_range=sigma_range, episode_lengths=episode_length,
                                                      limit_margin=limit_margin)
        monkeypatch.setattr(test_object, '_random_generator', MonkeyRandomGenerator())

        test_object._reference_value = reference_value
        self._monkey_get_current_value_counter = 0
        self._current_value = np.log10(sigma_range)
        test_object._reset_reference()
        assert sum(abs(test_object._reference - expected_reference)) < 1E-6, 'unexpected reference array'
        assert self._monkey_get_current_value_counter == 1, 'get_current_value() not called once'


class TestFurtherReferenceGenerator:
    """
    class for testing SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
    TriangularReferenceGenerator
    """
    # defined values for tests
    _kwargs = {}
    _physical_system = None
    _limit_margin = (0.0, 0.9)

    # counter
    _monkey_super_set_modules_counter = 0
    _monkey_get_current_value_counter = 0

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().__set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    def monkey_get_current_value(self, value):
        self._monkey_get_current_value_counter += 1
        return value

    @pytest.mark.parametrize('amplitude_range', [(0.1, 0.8)])
    @pytest.mark.parametrize('frequency_range', [(2, 150)])
    @pytest.mark.parametrize('offset_range', [(-0.8, 0.5)])
    @pytest.mark.parametrize('kwargs', [{}])
    @pytest.mark.parametrize("reference_class",
                             [SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
                              TriangularReferenceGenerator])
    def test_init(self, monkeypatch, reference_class, amplitude_range, frequency_range, offset_range, kwargs):
        """
        test initialization of different reference generators
        :param monkeypatch:
        :param reference_class: class name of tested reference generator
        :param amplitude_range: range of the amplitude
        :param frequency_range: range of the frequency
        :param offset_range: range of the offset
        :param kwargs: further arguments
        :return:
        """
        # setup test scenario

        self._kwargs = kwargs
        # call function to test
        test_object = reference_class(amplitude_range=amplitude_range, frequency_range=frequency_range,
                                      offset_range=offset_range, **kwargs)
        # verify the expected results
        assert test_object._amplitude_range == amplitude_range, 'amplitude range is not passed correctly'
        assert test_object._frequency_range == frequency_range, 'frequency range is not passed correctly'
        assert test_object._offset_range == offset_range, 'offset range is not passed correctly'

    @pytest.mark.parametrize("reference_class",
                             [SawtoothReferenceGenerator, SinusoidalReferenceGenerator, StepReferenceGenerator,
                              TriangularReferenceGenerator])
    @pytest.mark.parametrize('amplitude_range, expected_amplitude',
                             [((0.1, 0.8), (0.1, 0.45)), ((-0.5, 0.35), (0.0, 0.35))])
    @pytest.mark.parametrize('offset_range, expected_offset', [((-0.8, 0.5), (0.0, 0.5)), ((0.1, 0.96), (0.1, 0.9))])
    def test_set_modules(self, monkeypatch, reference_class, amplitude_range, offset_range,
                         expected_offset, expected_amplitude):
        """
        test set_modules()
        :param monkeypatch:
        :param reference_class: class name of tested reference generator
        :param amplitude_range: range of the amplitude
        :param offset_range: range of the offset
        :param expected_offset: expected result of the offset
        :param expected_amplitude: expected result of the amplitude
        :return:
        """
        # setup test scenario
        self._physical_system = DummyPhysicalSystem()
        monkeypatch.setattr(SubepisodedReferenceGenerator, 'set_modules', self.monkey_super_set_modules)
        test_object = reference_class(amplitude_range=amplitude_range, offset_range=offset_range)
        monkeypatch.setattr(test_object, '_limit_margin', self._limit_margin)
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert all(test_object._amplitude_range == expected_amplitude), 'amplitude range is not as expected'
        assert all(test_object._offset_range == expected_offset), 'offset range is not as expected'
        assert self._monkey_super_set_modules_counter == 1, 'super().set_modules() is not called once'

    @pytest.mark.parametrize('reference_class, expected_reference, frequency_range', [
        (SawtoothReferenceGenerator, np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, -0.4, -0.3, -0.2, -0.1]), 1 / 8),
        (SinusoidalReferenceGenerator,
         np.array([0.4,  0.2828427, 0.0, -0.2828427, -0.4, -0.2828427, 0.0, 0.2828427, 0.4, 0.2828427]), 1 / 8),
        (StepReferenceGenerator, np.array([0.4, -0.4, -0.4, -0.4, 0.4, 0.4, 0.4, -0.4, -0.4, -0.4]), 1/6),
        (TriangularReferenceGenerator, np.array([0.4, 0.2666667, 0.133333, 0.0, -0.133333, -0.2666667, -0.4, 0.0, 0.4,
                                                 0.2666667]), 1 / 8)])
    def test_reset_reference(self, monkeypatch, reference_class, expected_reference, frequency_range):
        # setup test scenario
        amplitude_range = 0.8
        offset_range = 0.5
        limit_margin = 0.4
        episode_length = 10
        dummy_physical_system = DummyPhysicalSystem()
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)

        class MonkeyRandomGenerator:
            _rg = DummyRandom()

            def uniform(self, mu=0, sigma=1):
                return self._rg.monkey_random_rand()

            def triangular(self, left=-1, mode=0, right=1):
                return self._rg.monkey_random_triangular(left, mode, right)

        test_object = reference_class(
            amplitude_range=amplitude_range, frequency_range=frequency_range,
            offset_range=offset_range, limit_margin=limit_margin,
            episode_lengths=episode_length, reference_state='dummy_state_0'
        )
        monkeypatch.setattr(test_object, '_random_generator', MonkeyRandomGenerator())
        test_object.set_modules(dummy_physical_system)
        # call function to test
        test_object._reset_reference()
        # verify expected results
        assert sum(abs(expected_reference - test_object._reference)) < 1E-6, 'unexpected reference'


class TestSubepisodedReferenceGenerator:
    """
    class to the SubepisodedReferenceGenerator
    """
    # defined values for tests
    _episode_length = (10, 50)
    _reference_state = 'dummy_1'
    _referenced_states = np.array([0, 1, 0])
    _referenced_states = _referenced_states.astype(bool)
    _value_range = None
    _current_value = 35
    _initial_state = None
    _physical_system = DummyPhysicalSystem()
    _state_names = ['dummy_0', 'dummy_1', 'dummy_2']
    _nominals = np.array([1, 2, 3])
    _limits = np.array([5, 7, 6])
    _reference = np.array([0, 1, 0])
    _reference_trajectory = np.array([0.4, 0, 0, 0, 0, 0, 0, 0, 0.6, 0, 0, 0, 0, 0])
    _state_space_low = np.zeros(3)
    _state_space_high = np.ones(3)
    _physical_system.state_space.low = _state_space_low
    _physical_system.state_space.high = _state_space_high
    _physical_system._limits = _limits
    _physical_system._nominal_values = _nominals
    _physical_system._state_names = _state_names
    _length_state = 3

    # counter
    _monkey_super_set_modules_counter = 0
    _monkey_super_reset_counter = 0
    _monkey_reset_reference_counter = 0

    def monkey_get_current_value(self, value_range):
        """
        mock function for get_current_value()
        :param value_range:
        :return:
        """
        assert value_range == self._value_range, 'value range is not as expected'
        return self._current_value

    def monkey_reset_reference(self):
        """
        mock function for reset_reference()
        :return:
        """
        self._monkey_reset_reference_counter += 1

    def monkey_super_reset(self, initial_state):
        """
        mock function for super().reset()
        :param initial_state:
        :return:
        """
        assert initial_state == self._initial_state, 'initial state is not as expected'
        self._monkey_super_reset_counter += 1
        return self._reference

    def monkey_state_array(self, input_values, state_names):
        """
        mock function for utils.set_state_array()
        :param input_values:
        :param state_names:
        :return:
        """
        assert input_values == {self._reference_state: 1}, \
            'the input values are not a dict with the reference state and value 1'
        assert state_names == self._state_names, 'state names are not as expected'
        return np.array([0, 1, 0])

    def monkey_super_set_modules(self, physical_system):
        """
        mock function for super().set_modules()
        :param physical_system:
        :return:
        """
        self._monkey_super_set_modules_counter += 1
        assert physical_system == self._physical_system, 'physical system is not the expected instance'

    @pytest.mark.parametrize("limit_margin", [None, 0.3, (-0.1, 0.8)])
    def test_init(self, monkeypatch, limit_margin):
        """
        test __init__()
        :param monkeypatch:
        :param limit_margin: possible values for limit margin
        :return:
        """
        # setup test scenario
        self._value_range = self._episode_length

        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        # call function to test
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state,
                                                    episode_lengths=self._episode_length, limit_margin=limit_margin)
        # verify the expected results
        assert test_object._limit_margin == limit_margin, 'limit margin is not passed correctly'
        assert test_object._reference_value == 0.0, 'the reference value is not 0'
        assert test_object._reference_state == self._reference_state, 'reference state is not passed correctly'
        assert test_object._episode_len_range == self._episode_length, 'episode length is not passed correctly'
        assert test_object._current_episode_length == self._current_value, 'current episode length is not as expected'
        assert test_object._k == 0, 'current reference step is not 0'

    @pytest.mark.parametrize("limit_margin, expected_low, expected_high",
                             [(None, 0, 2 / 7), (0.3, 0.0, 0.3), ((-0.1, 0.8), 0, 0.8)])
    def test_set_modules(self, monkeypatch, limit_margin, expected_low, expected_high):
        """
        test set_modules()
        :param monkeypatch:
        :param limit_margin: possible values for limit margin
        :param expected_low: expected value for lower limit margin
        :param expected_high: expected value for upper limit margin
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(ReferenceGenerator, 'set_modules', self.monkey_super_set_modules)
        monkeypatch.setattr(srg, 'set_state_array', self.monkey_state_array)
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state, limit_margin=limit_margin)
        # call function to test
        test_object.set_modules(self._physical_system)
        # verify the expected results
        assert self._monkey_super_set_modules_counter == 1, 'super().set_modules() is not called once'
        assert all(test_object.reference_space.low == expected_low), 'lower reference space not as expected'
        assert all(test_object.reference_space.high == expected_high), 'upper reference space not as expected'
        test_object._limit_margin = ['test']
        # call function to test
        with pytest.raises(Exception):
            test_object.set_modules(self._physical_system)

    @pytest.mark.parametrize('initial_state', [None, _initial_state])
    @pytest.mark.parametrize('initial_reference, expected_reference', [(np.array([0.2, 0.4, 0.7]), 0.4), (None, 0.0)])
    def test_reset(self, monkeypatch, initial_reference, initial_state, expected_reference):
        """
        test reset()
        :param monkeypatch:
        :param initial_reference: possible values for the initial reference
        :param initial_state: possible values for the initial state
        :param expected_reference: expected value for the reference
        :return:
        """
        # setup test scenario
        monkeypatch.setattr(ReferenceGenerator, 'reset', self.monkey_super_reset)
        test_object = SubepisodedReferenceGenerator(reference_state=self._reference_state)
        monkeypatch.setattr(test_object, '_referenced_states', self._referenced_states)
        # call function to test
        reference = test_object.reset(initial_state, initial_reference)
        # verify the expected results
        assert all(reference == self._reference), 'reference not as expected'
        assert test_object._current_episode_length == -1, 'current episode length is not -1'
        assert test_object._reference_value == expected_reference, 'unexpected reference value'
        assert self._monkey_super_reset_counter == 1, 'super().reset() not called once'

    def test_get_reference(self, monkeypatch):
        """
        test get_reference()
        :param monkeypatch:
        :return:
        """
        # setup test scenario
        test_object = SubepisodedReferenceGenerator()
        monkeypatch.setattr(test_object, '_referenced_states', self._referenced_states)
        monkeypatch.setattr(test_object, '_reference_value', 0.4)
        # call function to test
        reference = test_object.get_reference()
        # verify the expected results
        assert all(reference == np.array([0, 0.4, 0])), 'unexpected reference'

    @pytest.mark.parametrize('k, expected_reference, expected_parameter', [(8, 0.6, (10, 0)), (10, 0.4, (35, 1))])
    # setup test scenario
    def test_get_reference_observation(self, monkeypatch, k, expected_reference, expected_parameter):
        """
        test get_reference_observation()
        :param monkeypatch:
        :param k: current time step for testing
        :param expected_reference: expected value for the reference
        :param expected_parameter: expected counter for reset reference
        :return:
        """
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_get_current_value', self.monkey_get_current_value)
        monkeypatch.setattr(SubepisodedReferenceGenerator, '_reset_reference', self.monkey_reset_reference)
        self._value_range = self._episode_length
        test_object = SubepisodedReferenceGenerator(episode_lengths=self._episode_length)
        monkeypatch.setattr(test_object, '_reference', self._reference_trajectory)
        monkeypatch.setattr(test_object, '_k', k)
        monkeypatch.setattr(test_object, '_current_episode_length', 10)
        # call function to test
        reference = test_object.get_reference_observation()
        # verify the expected results
        assert reference == np.array([expected_reference]), 'unexpected reference'
        assert test_object._current_episode_length == expected_parameter[0], 'unexpected current episode length'
        assert self._monkey_reset_reference_counter == expected_parameter[1], \
            'unexpected number of calls of reset_reference, depends on the setting'

    @pytest.mark.parametrize(
        'value_range, expected_value',
        [(12, 12), (1.0365, 1.0365), ([0, 1.6], 0.4), ((-0.12, 0.4), 0.01), (np.array([-0.5, 0.6]), -0.225)]
    )
    def test_get_current_value(self, monkeypatch, value_range, expected_value):
        # setup test scenario

        # call function to test
        test_object = SubepisodedReferenceGenerator()

        class MonkeyUniformGenerator:
            _rg = DummyRandom()

            def uniform(self, mu=0, sigma=1):
                return self._rg.monkey_random_rand()

        monkeypatch.setattr(test_object, '_random_generator', MonkeyUniformGenerator())
        val = test_object._get_current_value(value_range)
        # verify expected results
        assert abs(val - expected_value) < 1E-6, 'unexpected value from get_current_value().'


class TestConstReferenceGenerator(TestReferenceGenerator):

    key = 'ConstReference'
    class_to_test = ConstReferenceGenerator

    @pytest.fixture
    def physical_system(self):
        return DummyPhysicalSystem(state_length=2)

    @pytest.fixture
    def reference_generator(self, physical_system):
        rg = ConstReferenceGenerator(reference_state=physical_system.state_names[1], reference_value=1)
        rg.set_modules(physical_system)
        return rg

    @pytest.mark.parametrize('reference_value, reference_state', [(0.8, 'abc')])
    def test_initialization(self, reference_value, reference_state):
        rg = ConstReferenceGenerator(reference_value=reference_value, reference_state=reference_state)
        assert rg.reference_space == Box(np.array([reference_value]), np.array([reference_value]))
        assert rg._reference_value == reference_value
        assert rg._reference_state == reference_state

    def test_set_modules(self, physical_system):
        rg = ConstReferenceGenerator(reference_state=physical_system.state_names[1])
        rg.set_modules(physical_system)
        assert all(rg._referenced_states == np.array([False, True]))

    def test_get_reference(self, reference_generator):
        assert all(reference_generator.get_reference() == np.array([0, 1]))

    def test_get_reference_observation(self, reference_generator):
        assert all(reference_generator.get_reference_observation() == np.array([1]))

    def test_registered(self):
        if self.key != '':
            rg = gem.utils.instantiate(ReferenceGenerator, self.key)
            assert type(rg) == self.class_to_test


class TestMultipleReferenceGenerator(TestReferenceGenerator):

    key = 'MultipleReference'
    class_to_test = MultipleReferenceGenerator

    @pytest.fixture
    def reference_generator(self):
        physical_system = DummyPhysicalSystem(state_length=3)
        sub_generator_1 = DummyReferenceGenerator(reference_state='dummy_state_0')
        sub_generator_2 = DummyReferenceGenerator(reference_state='dummy_state_1')
        rg = self.class_to_test([sub_generator_1, sub_generator_2])
        sub_generator_1._referenced_states = np.array([False, False, True])
        sub_generator_2._referenced_states = np.array([False, True, False])
        rg.set_modules(physical_system)
        sub_generator_1.reference_array = np.array([0, 0, 1])
        sub_generator_2.reference_array = np.array([0, 1, 0])
        sub_generator_1.reference_observation = np.array([-1])
        sub_generator_2.reference_observation = np.array([1])
        return rg

    def test_set_modules(self):
        physical_system = DummyPhysicalSystem(state_length=3)
        sub_generator_1 = DummyReferenceGenerator(reference_state='dummy_state_0')
        sub_generator_2 = DummyReferenceGenerator(reference_state='dummy_state_1')
        rg = self.class_to_test([sub_generator_1, sub_generator_2])
        sub_generator_1._referenced_states = np.array([False, False, True])
        sub_generator_2._referenced_states = np.array([False, True, False])
        rg.set_modules(physical_system)
        assert sub_generator_1.physical_system == physical_system
        assert sub_generator_2.physical_system == physical_system
        assert all(rg.reference_space.low == np.array([0, 0]))
        assert all(rg.reference_space.high == np.array([1, 1]))
        assert all(rg.referenced_states == np.array([1, 1, 0]))

    def test_reset(self, reference_generator):
        ref, ref_obs, _ = reference_generator.reset()
        assert all(ref == np.array([0, 1, 1]))
        assert all(ref_obs == np.array([-1, 1]))

    def test_get_reference(self, reference_generator):
        assert all(reference_generator.get_reference(0) == np.array([0, 1, 1]))

    def test_reference_observation(self,reference_generator):
        assert all(reference_generator.get_reference_observation(0) == np.array([-1, 1]))

    def test_registered(self):
        if self.key != '':
            rg = gem.utils.instantiate(ReferenceGenerator, self.key, sub_generators=[DummyReferenceGenerator])
            assert type(rg) == self.class_to_test

# endregion

