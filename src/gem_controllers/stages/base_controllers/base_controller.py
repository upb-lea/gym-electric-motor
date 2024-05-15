from ..stage import Stage
from .e_base_controller_task import EBaseControllerTask


class BaseController(Stage):
    """The base controller is the base class for all dynamic control stages like the P-I-D controllers or the
    Three-Point controller.

    In contrast to other stages, the base controllers can be used for multiple tasks e.g. a speed control with a
    reference as output or a current control with voltages as output.
    """

    def __init__(self, control_task):
        """
        Args:
            control_task(str): Control task of the base controller.
        """
        self._control_task = EBaseControllerTask.get_control_task(control_task)

    def __call__(self, state, reference):
        """
        Calculate the reference value of the base controller.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference of the state.

        Returns:
            np.array: reference values of the next stage
        """
        raise NotImplementedError

    def tune(self, env, env_id, **base_controller_kwargs):
        """
        Tune a base controller.

        Args:
             env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
             env_id(str): The corresponding environment-id to specify the concrete environment.
             **base_controller_kwargs: Keyword arguments, that should be passed to a base controller
        """
        pass

    def feedback(self, state, reference, clipped_reference):
        pass
