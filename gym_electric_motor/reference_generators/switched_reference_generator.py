import numpy as np
from gym.spaces import Box

from ..random_component import RandomComponent
from ..core import ReferenceGenerator
from ..utils import instantiate


class SwitchedReferenceGenerator(ReferenceGenerator, RandomComponent):
    """Reference Generator that switches randomly between multiple sub generators with a certain probability p for each.
    """

    def __init__(self, sub_generators, p=None, super_episode_length=(100, 10000)):
        """
        Args:
            sub_generators(list(ReferenceGenerator)): ReferenceGenerator instances to be used as the sub_generators.
            p(list(float)/None): (Optional) Probabilities for each sub_generator. If None a uniform
                probability for each sub_generator is used.
            super_episode_length(Tuple(int, int)): Minimum and maximum number of time steps a sub_generator is used.
        """
        ReferenceGenerator.__init__(self)
        RandomComponent.__init__(self)
        self.reference_space = Box(-1, 1, shape=(1,), dtype=np.float64)
        self._reference = None
        self._k = 0

        self._sub_generators = list(sub_generators)
        assert len(self._sub_generators) > 0, 'No sub generator was passed.'
        ref_names = self._sub_generators[0].reference_names
        assert all(sub_gen.reference_names == ref_names for sub_gen in self._sub_generators),\
            'The passed sub generators have different referenced states.'
        self._reference_names = ref_names
        self._probabilities = p or [1/len(sub_generators)] * len(sub_generators)
        self._current_episode_length = 0
        if type(super_episode_length) in [float, int]:
            super_episode_length = super_episode_length, super_episode_length + 1
        self._super_episode_length = super_episode_length
        self._current_ref_generator = self._sub_generators[0]

    def set_modules(self, physical_system):
        """
        Args:
            physical_system(PhysicalSystem): The physical system of the environment.
        """
        super().set_modules(physical_system)
        for sub_generator in self._sub_generators:
            sub_generator.set_modules(physical_system)
        ref_space_low = np.min([sub_generator.reference_space.low for sub_generator in self._sub_generators], axis=0)
        ref_space_high = np.max([sub_generator.reference_space.high for sub_generator in self._sub_generators], axis=0)
        self.reference_space = Box(ref_space_low, ref_space_high)
        self._referenced_states = self._sub_generators[0].referenced_states
        for sub_generator in self._sub_generators:
            assert np.all(sub_generator.referenced_states == self._referenced_states), \
                'Reference Generators reference different state variables'
            assert sub_generator.reference_space.shape == self.reference_space.shape, \
                'Reference Generators have differently shaped reference spaces'

    def reset(self, initial_state=None, initial_reference=None):
        self.next_generator()
        self._reset_reference()
        return self._current_ref_generator.reset(initial_state, initial_reference)

    def get_reference(self, state, **kwargs):
        self._reference = self._current_ref_generator.get_reference(state, **kwargs)
        return self._reference

    def get_reference_observation(self, state, *_, **kwargs):
        if self._k >= self._current_episode_length:
            self._reset_reference()
            _, obs, _ = self._current_ref_generator.reset(state, self._reference)
        else:
            obs = self._current_ref_generator.get_reference_observation(state, **kwargs)
        self._k += 1
        return obs

    def _reset_reference(self):
        self._current_episode_length = self.random_generator.integers(
            self._super_episode_length[0], self._super_episode_length[1]
        )
        self._k = 0
        self._current_ref_generator = self.random_generator.choice(self._sub_generators, p=self._probabilities)

    def seed(self, seed=None):
        super().seed(seed)
        for sub_generator in self._sub_generators:
            if isinstance(sub_generator, RandomComponent):
                seed = self._seed_sequence.spawn(1)[0]
                sub_generator.seed(seed)
