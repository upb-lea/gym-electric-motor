import numpy as np
import gem_controllers as gc


class CurrentController(gc.GemController):
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
