import numpy as np

from .emf_feedforward import EMFFeedforward


class EMFFeedforwardInd(EMFFeedforward):
    """
    This class extends the functions of the EMFFeedforward class to decouple the dq-components of the induction motor.
    """

    def __init__(self):
        super().__init__()
        self.r_r = None
        self.l_m = None
        self.l_r = None
        self.i_sq_idx = None
        self.psi_idx = None
        self.psi_abs_idx = None

    def __call__(self, state, reference):
        """
        Decouple the input voltages of an induction motor.

        Args:
             state(np.ndarray): The state of the environment.
             reference(np.ndarray): The reference voltages.

        Returns:
            np.array: decoupled reference voltages
        """

        # Calculate the stator angular velocity
        omega_s = state[self.omega_idx] + self.r_r * self.l_m / self.l_r * state[self.i_sq_idx] / max(
            np.abs(state[self.psi_abs_idx]), 1e-4
        ) * np.sign(state[self.psi_abs_idx])

        # Calculate the decoupling of the components
        action = (
            reference
            + omega_s * self.inductance * state[self.current_indices]
            + np.array([-self.l_m * self.r_r / (self.l_r**2), state[self.omega_idx] * self.l_m / self.l_r])
            * state[self.psi_abs_idx]
        )

        return action

    def tune(self, env, env_id, **_):
        """
        Set all needed motor parameters for the decoupling.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """

        super().tune(env, env_id, **_)
        mp = env.physical_system.electrical_motor.motor_parameter
        self.r_r = mp["r_r"]
        self.l_m = mp["l_m"]
        self.l_r = mp["l_sigr"] + self.l_m
        self.i_sq_idx = env.state_names.index("i_sq")
        self.psi_abs_idx = env.state_names.index("psi_abs")
