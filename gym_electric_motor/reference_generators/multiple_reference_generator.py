import numpy as np
from gym.spaces import Box

from ..core import ReferenceGenerator
from ..utils import instantiate


class MultipleReferenceGenerator(ReferenceGenerator):
    """
    Reference Generator that combines multiple sub reference generators that all have to reference
    different state variables.
    """

    reference_space = Box(-1, 1, shape=(1,))

    def __init__(self, sub_generators, sub_args=None, **kwargs):
        """
        Args:
            sub_generators(list(str/class/object)): List of keys, classes or objects to instantiate the sub_generators
            sub_args(dict/list(dict)/None): (Optional) Arguments to pass to the sub_converters. If not passed all kwargs
                will be passed to each sub_generator.
            kwargs: All kwargs of the environment. Passed to the sub_generators, if no sub_args are passed.
        """
        super().__init__()
        if type(sub_args) is dict:
            sub_arguments = [sub_args] * len(sub_generators)
        elif hasattr(sub_args, '__iter__'):
            assert len(sub_args) == len(sub_generators)
            sub_arguments = sub_args
        else:
            sub_arguments = [kwargs] * len(sub_generators)
        self._sub_generators = [instantiate(ReferenceGenerator, sub_generator, **sub_arg)
                                for sub_generator, sub_arg in zip(sub_generators, sub_arguments)]

    def set_modules(self, physical_system):
        """
        Args:
            physical_system(PhysicalSystem): The physical system of the environment.
        """
        super().set_modules(physical_system)
        for sub_generator in self._sub_generators:
            sub_generator.set_modules(physical_system)

        # Ensure that all referenced states are different
        assert all(sum([sub_generator.referenced_states.astype(int) for sub_generator in self._sub_generators]) < 2), \
            'Some of the passed reference generators share the same reference variable'

        ref_space_low = np.concatenate([sub_generator.reference_space.low for sub_generator in self._sub_generators])
        ref_space_high = np.concatenate([sub_generator.reference_space.high for sub_generator in self._sub_generators])
        self.reference_space = Box(ref_space_low, ref_space_high)
        self._referenced_states = np.sum(
            [sub_generator.referenced_states for sub_generator in self._sub_generators], dtype=bool, axis=0
        )

    def reset(self, initial_state=None, initial_reference=None):
        # docstring from superclass
        refs = np.zeros_like(self._physical_system.state_names, dtype=float)
        ref_obs = np.array([])
        for sub_generator in self._sub_generators:
            ref, ref_observation, _ = sub_generator.reset(initial_state, initial_reference)
            refs += ref
            ref_obs = np.concatenate((ref_obs, ref_observation))
        return refs, ref_obs, None

    def get_reference(self, state, **kwargs):
        # docstring from superclass
        return sum([sub_generator.get_reference(state, **kwargs) for sub_generator in self._sub_generators])

    def get_reference_observation(self, state, *_, **kwargs):
        # docstring from superclass
        return np.concatenate(
            [sub_generator.get_reference_observation(state, **kwargs) for sub_generator in self._sub_generators]
        )
