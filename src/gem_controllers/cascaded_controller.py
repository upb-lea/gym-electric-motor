from .gem_controller import GemController


class CascadedController(GemController):
    """The CascadedController class contains a controller with multiple hierarchically structured stages."""

    def control(self, state, reference):
        """
        The function iterates through all the stages to calculate the action.

        Args:
            state(np.array): Array that contains the actual state of the environment
            reference(np.array): Array that contains the actual references of the referenced states

        Returns:
            action
        """

        for stage in self._stages:
            reference = stage(state, reference)
        return reference
