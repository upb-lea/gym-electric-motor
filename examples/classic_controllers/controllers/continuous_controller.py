class ContinuousController:
    """The class ContinuousController is the base for all continuous base controller (P-I-D-Controller)"""

    @classmethod
    def make(cls, environment, stage, _controllers, **controller_kwargs):
        controller = _controllers[stage["controller_type"]][2](
            environment, param_dict=stage, **controller_kwargs
        )
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass


class PController(ContinuousController):
    def __init__(self, p_gain=5):
        self.p_gain = p_gain


class IController(ContinuousController):
    def __init__(self, i_gain=10):
        self.i_gain = i_gain
        self.integrated = 0

    def integrate(self, state, reference):
        self.integrated += (reference - state) * self.tau


class DController(ContinuousController):
    def __init__(self, d_gain=0.05):
        self.d_gain = d_gain
        self.e_old = 0
