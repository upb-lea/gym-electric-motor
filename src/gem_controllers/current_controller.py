import numpy as np

from .gem_controller import GemController


class CurrentController(GemController):
    """Base class for a current controller"""

    def control(self, state, reference):
        raise NotImplementedError

    def tune(self, env, env_id, **kwargs):
        raise NotImplementedError

    @property
    def voltage_reference(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def t_n(self) -> np.ndarray:
        raise NotImplementedError
