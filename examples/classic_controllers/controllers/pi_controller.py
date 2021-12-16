from .continuous_controller import PController, IController


class PIController(PController, IController):
    """
        The PI-Controller is a combination of the base P-Controller and the base I-Controller. The integrate function is
        executed after checking compliance with the limitations in the higher-level controller stage in order to adjust
        the I-component of the controller accordingly.
    """

    def __init__(self, environment, p_gain=5, i_gain=5, param_dict={}, **controller_kwargs):
        self.tau = environment.physical_system.tau

        p_gain = param_dict.get('p_gain', p_gain)
        i_gain = param_dict.get('i_gain', i_gain)
        PController.__init__(self, p_gain)
        IController.__init__(self, i_gain)

    def control(self, state, reference):
        return self.p_gain * (reference - state) + self.i_gain * (self.integrated + (reference - state) * self.tau)

    def reset(self):
        self.integrated = 0
