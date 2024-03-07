from gymnasium.spaces import Discrete, MultiDiscrete


class DiscreteController:
    """
    The DiscreteController is the base class for the base discrete controllers (OnOff controller and three-point
    controller).
    """

    @classmethod
    def make(cls, environment, stage, _controllers, **controller_kwargs):
        if type(environment.action_space) is Discrete:
            action_space_n = environment.action_space.n
        elif type(environment.action_space) is MultiDiscrete:
            action_space_n = environment.action_space.nvec[0]
        else:
            action_space_n = 3

        controller = _controllers[stage["controller_type"]][2](
            environment,
            action_space=action_space_n,
            param_dict=stage,
            **controller_kwargs,
        )
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass
