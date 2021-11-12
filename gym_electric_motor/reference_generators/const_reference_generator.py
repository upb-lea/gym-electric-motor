import numpy as np
from gym_electric_motor.core import ReferenceGenerator
from gym_electric_motor.utils import set_state_array
from gym.spaces import Box


class ConstReferenceGenerator(ReferenceGenerator):
    """
    Reference Generator that generates a constant reference for a single state variable.
    """

    def __init__(self, reference_state='omega', reference_value=0.5, **kwargs):
        """
        Args:
            reference_value(float): Normalized Value for the const reference.
            reference_state(string): Name of the state to reference
            kwargs(dict): Arguments passed to the superclass ReferenceGenerator.
        """
        super().__init__(**kwargs)
        self._reference_value = reference_value
        self._reference_state = reference_state.lower()
        self.reference_space = Box(np.array([reference_value]), np.array([reference_value]), dtype=np.float64)
        self._reference_names = self._reference_state

    def set_modules(self, physical_system):
        # docstring from superclass
        super().set_modules(physical_system)
        self._referenced_states = set_state_array(
            {self._reference_state: 1}, physical_system.state_names
        ).astype(bool)

    def get_reference(self, *_, **__):
        # docstring from superclass
        reference = np.zeros_like(self._referenced_states, dtype=float)
        reference[self._referenced_states] = self._reference_value
        return reference

    def get_reference_observation(self, *_, **__):
        # docstring from superclass
        return np.array([self._reference_value])
