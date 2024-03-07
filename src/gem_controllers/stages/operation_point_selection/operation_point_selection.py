from gem_controllers.stages.stage import Stage


class OperationPointSelection(Stage):
    """Base class for all operation point selections."""

    def __call__(self, state, reference):
        """
        Calculate the current operation point for a given torque reference value.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            current_reference(np.ndarray): references for the current control stage

        """
        return self._select_operating_point(state, reference)

    def _select_operating_point(self, state, reference):
        """
        Interal calculation of the operation point

        Args:
            state(np.ndarray): The state of the environment.
            reference(np.ndarray): The reference of the state.

        Returns:
            np.array: current refernce values
        """

        raise NotImplementedError

    def tune(self, env, env_id, current_safety_margin=0.2):
        """
        Set the motor parameters, limits and indices of a operation point selection class.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            current_safety_margin(float): Percentage of the current margin to the current limit.
        """
        pass
