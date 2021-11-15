from .discrete_controller import DiscreteController


class OnOffController(DiscreteController):
    """This is a hysteresis controller with two possible output states."""

    def __init__(self, environment, action_space, hysteresis=0.02, param_dict={}, cascaded=False, control_e=False,
                 **controller_kwargs):
        self.hysteresis = param_dict.get('hysteresis', hysteresis)
        self.switch_on_level = 1

        self.switch_off_level = 2 if action_space in [3, 4] and not control_e else 0
        if cascaded:
            self.switch_off_level = int(environment.physical_system.state_space.low[0])

        self.action = self.switch_on_level

    def control(self, state, reference):
        if reference - state > self.hysteresis:
            self.action = self.switch_on_level

        elif reference - state < self.hysteresis:
            self.action = self.switch_off_level

        return self.action

    def reset(self):
        self.action = self.switch_on_level
