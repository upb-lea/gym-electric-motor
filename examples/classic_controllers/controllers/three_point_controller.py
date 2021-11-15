from .discrete_controller import DiscreteController


class ThreePointController(DiscreteController):
    """This is a hysteresis controller with three possible output states."""

    def __init__(self, environment, action_space, switch_to_positive_level=0.02, switch_to_negative_level=0.02,
                 switch_to_neutral_from_positive=0.01, switch_to_neutral_from_negative=0.01, param_dict={},
                 cascaded=False, control_e=False, **controller_kwargs):

        self.pos = param_dict.get('switch_to_positive_level', switch_to_positive_level)
        self.neg = param_dict.get('switch_to_negative_level', switch_to_negative_level)
        self.neutral_from_pos = param_dict.get('switch_to_neutral_from_positive', switch_to_neutral_from_positive)
        self.neutral_from_neg = param_dict.get('switch_to_neutral_from_negative', switch_to_neutral_from_negative)

        self.negative = 2 if action_space in [3, 4, 8] and not control_e else 0
        if cascaded:
            self.negative = int(environment.physical_system.state_space.low[0])
        self.positive = 1
        self.neutral = 0

        self.action = self.neutral
        self.recent_action = self.neutral

    def control(self, state, reference):
        if reference - state > self.pos or ((self.neutral_from_pos < reference - state) and self.recent_action == 1):
            self.action = self.positive
            self.recent_action = 1
        elif reference - state < -self.neg or (
                (-self.neutral_from_neg > reference - state) and self.recent_action == 2):
            self.action = self.negative
            self.recent_action = 2
        else:
            self.action = self.neutral
            self.recent_action = 0

        return self.action

    def reset(self):
        self.action = self.neutral
        self.recent_action = self.neutral