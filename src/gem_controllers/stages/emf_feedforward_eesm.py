import numpy as np

from .emf_feedforward import EMFFeedforward


class EMFFeedforwardEESM(EMFFeedforward):
    """
    This class extends the functions of the EMFFeedforward class to decouple the dq-components of the induction motor.
    """

    def __init__(self):
        super().__init__()
        self._l_m = None
        self._i_e_idx = None
        self._decoupling_params = None
        self._action_decoupling = None
        self._currents_idx = None
        self._action_idx = None

    def __call__(self, state, reference):
        """
        Calculate the emf feedforward voltages and add them to the actions of the current controller.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference voltages.

        Returns:
            input voltages(np.ndarray): decoupled input voltages
        """

        self.psi = np.array([0, self._l_m * state[self._i_e_idx], 0])
        action = super().__call__(state, reference)
        action = action + self._decoupling_params * state[self._currents_idx]
        action = action + self._action_decoupling * action[self._action_idx]
        return action

    def tune(self, env, env_id, **_):
        """
        Set all needed motor parameters for the decoupling.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        super().tune(env, env_id, **_)
        l_m = env.physical_system.electrical_motor.motor_parameter["l_m"]
        l_d = env.physical_system.electrical_motor.motor_parameter["l_d"]
        l_e = env.physical_system.electrical_motor.motor_parameter["l_e"]
        r_s = env.physical_system.electrical_motor.motor_parameter["r_s"]
        r_e = env.physical_system.electrical_motor.motor_parameter["r_e"]

        self._l_m = l_m
        self._i_e_idx = env.state_names.index("i_e")
        self._decoupling_params = np.array([-l_m * r_e / l_e, 0, -l_m * r_s / l_d])
        self._action_decoupling = np.array([l_m / l_e, 0, l_m / l_d])
        self._currents_idx = [env.state_names.index("i_e"), 0, env.state_names.index("i_sd")]
        self._action_idx = [2, 1, 0]
