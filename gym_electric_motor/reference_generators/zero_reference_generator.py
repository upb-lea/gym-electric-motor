import numpy as np
from gym.spaces import Box

from ..core import ReferenceGenerator


class ZeroReferenceGenerator(ReferenceGenerator):
    """Dummy Reference Generator that does not generate any reference but zeros for all states."""

    def __init__(self):
        super().__init__()
        self.reference_space = Box(0, 0, (0,))
        self._reference_names = []

    def set_modules(self, physical_system):
        super().set_modules(physical_system)
        self._referenced_states = np.zeros_like(self._physical_system.state_names, dtype=bool)

    def get_reference(self, state=None, *_, **__):
        return np.zeros_like(self._physical_system.state_names, dtype=float)

    def get_reference_observation(self, state=None, *_, **__):
        return np.array([])
