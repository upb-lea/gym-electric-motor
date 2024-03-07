import numpy as np

import gem_controllers as gc
from gym_electric_motor.physical_systems.electric_motors import SynchronousMotor

from .. import parameter_reader as reader
from .stage import Stage


class AbcTransformation(Stage):
    """This class calculates the transformation from the dq-coordinate system to the abc-coordinatesystem for
    three-phase motors. Optionally, an advanced factor can be added to the angle to take the dead time of the inverter
    and the sampling time into account.
    """

    @property
    def advance_factor(self):
        """Advance factor of the angle."""
        return self._advance_factor

    @advance_factor.setter
    def advance_factor(self, value):
        self._advance_factor = float(value)

    @property
    def tau(self):
        """Sampling time of the system."""
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = float(value)

    def __init__(self):
        super().__init__()
        self._tau = 1e-4
        self._advance_factor = 0.5
        self.omega_idx = None
        self.angle_idx = None
        self._output_len = None

    def __call__(self, state, reference):
        """
        Args:
            state(np.array): state of the environment
            reference(np.array): voltage reference values

        Returns:
            np.array: reference values for the input voltages
        """

        epsilon_adv = self._angle_advance(state)  # calculate the advance angle
        output = np.zeros(self._output_len)
        output[0:3] = SynchronousMotor.t_32(SynchronousMotor.q(reference[0:2], epsilon_adv))
        if self._output_len > 3:
            output[3:] = reference[2:]
        return output

    def _angle_advance(self, state):
        """Multiply the advance factor with the speed and the sampling time to calculate the advance angle"""
        return state[self.angle_idx] + self._advance_factor * self.tau * state[self.omega_idx]

    def tune(self, env, env_id, **_):
        """
        Tune the advance factor of the transformation.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
        """
        if gc.utils.get_motor_type(env_id) in gc.parameter_reader.induction_motors:
            self.angle_idx = env.state_names.index("psi_angle")
        else:
            self.angle_idx = env.state_names.index("epsilon")
        self.omega_idx = env.state_names.index("omega")
        if hasattr(env.physical_system.converter, "dead_time"):
            self._advance_factor = 1.5 if env.physical_system.converter.dead_time else 0.5
        else:
            self._advance_factor = 0.5
        action_type, _, motor_type = gc.utils.split_env_id(env_id)
        self._output_len = len(reader.get_output_voltages(motor_type, action_type))
