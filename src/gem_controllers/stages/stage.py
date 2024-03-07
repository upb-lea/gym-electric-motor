class Stage:
    """The stage is the basic module in the gem-controller structure."""

    def __call__(self, state, reference):
        """The stages control function.

        Args:
            state(numpy.ndarray): The denormalized state of the environment.
            reference(numpy.ndarray): The actual reference value for this stage.
        Returns:
            numpy.ndarray: The new reference-value for the next state.
        """
        raise NotImplementedError

    def reset(self):
        """Resets the stage to an initial state (e.g. before a new episode starts)."""
        pass

    def tune(self, env, env_id, **kwargs):
        """Fits the stages parameters to the passed environment.

        Args:
            env(gym_electric_motor.ElectricMotorEnvironment): The environment to be controlled.
            env_id(str): The id of the environment.
            **kwargs(dict): Optional further parameters to tune the stages.
        """
        pass
