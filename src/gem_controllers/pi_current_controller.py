import numpy as np

import gem_controllers as gc


class PICurrentController(gc.CurrentController):
    """This class forms the PI current controller, for any motor."""

    @property
    def signal_names(self):
        """Signal names of the calculated values."""
        return ["u_PI", "u_ff", "u_out"]

    @property
    def transformation_stage(self):
        """Coordinate transformation stage at the output"""
        return self._transformation_stage

    @property
    def current_base_controller(self) -> gc.stages.BaseController:
        """Base controller for the current control"""
        return self._current_base_controller

    @current_base_controller.setter
    def current_base_controller(self, value: gc.stages.BaseController):
        assert isinstance(value, gc.stages.BaseController)
        self._current_base_controller = value

    @property
    def emf_feedforward(self) -> gc.stages.EMFFeedforward:
        """EMF feedforward stage of the current controller"""
        return self._emf_feedforward

    @emf_feedforward.setter
    def emf_feedforward(self, value: gc.stages.EMFFeedforward):
        assert isinstance(value, gc.stages.EMFFeedforward)
        self._emf_feedforward = value

    @property
    def stages(self):
        """List of the stages up to the current controller"""
        stages_ = [self._current_base_controller]
        if self._decoupling:
            stages_.append(self._emf_feedforward)
        if self._coordinate_transformation_required:
            stages_.append(self._transformation_stage)
        stages_.append(self._clipping_stage)
        return stages_

    @property
    def voltage_reference(self) -> np.ndarray:
        """Reference values for the input voltages"""
        return self._voltage_reference

    @property
    def clipping_stage(self):
        """Clipping stage of the current controller"""
        return self._clipping_stage

    @property
    def t_n(self):
        """Time constant of the current controller"""
        if hasattr(self._current_base_controller, "p_gain") and hasattr(self._current_base_controller, "i_gain"):
            return self._current_base_controller.p_gain / self._current_base_controller.i_gain
        else:
            return self._tau_current_loop

    @property
    def references(self):
        """Reference values of the current control stage."""
        return dict()

    @property
    def referenced_states(self):
        """Referenced states of the current control stage."""
        return np.array([])

    @property
    def maximum_reference(self):
        return dict()

    def __init__(self, env, env_id, base_current_controller="PI", decoupling=True):
        """
        Initilizes a PI current control stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            base_current_controller(str): Selection which base controller should be used for the current control stage.
            decoupling(bool): Flag, if a EMF-Feedforward correction stage should be used in the PI current controller.
        """

        super().__init__()
        self._current_base_controller = None
        self._emf_feedforward = None
        self._transformation_stage = None
        self._tau_current_loop = np.array([0.0])
        self._coordinate_transformation_required = False
        self._decoupling = decoupling
        self._voltage_reference = np.array([])
        self._transformation_stage = gc.stages.AbcTransformation()

        # Choose the emf feedforward function
        if gc.utils.get_motor_type(env_id) in gc.parameter_reader.induction_motors:
            self._emf_feedforward = gc.stages.EMFFeedforwardInd()
        elif gc.utils.get_motor_type(env_id) == "EESM":
            self._emf_feedforward = gc.stages.EMFFeedforwardEESM()
        else:
            self._emf_feedforward = gc.stages.EMFFeedforward()

        # Choose the clipping function
        if gc.utils.get_motor_type(env_id) == "EESM":
            self._clipping_stage = gc.stages.clipping_stages.CombinedClippingStage("CC")
        elif gc.utils.get_motor_type(env_id) in gc.parameter_reader.ac_motors:
            self._clipping_stage = gc.stages.clipping_stages.SquaredClippingStage("CC")
        else:
            self._clipping_stage = gc.stages.clipping_stages.AbsoluteClippingStage("CC")
        self._anti_windup_stage = gc.stages.AntiWindup("CC")
        self._current_base_controller = gc.stages.base_controllers.get(base_current_controller)("CC")

    def tune(self, env, env_id, a=4, **kwargs):
        """
        Tune the components of the current control stage.

        Args:
            env(ElectricMotorEnvironment): The GEM-Environment that the controller shall be created for.
            env_id(str): The corresponding environment-id to specify the concrete environment.
            a(float): Design parameter of the symmetric optimum for the base controllers
        """

        action_type = gc.utils.get_action_type(env_id)
        motor_type = gc.utils.get_motor_type(env_id)
        if action_type in ["Finite", "Cont"] and motor_type in gc.parameter_reader.ac_motors:
            self._coordinate_transformation_required = True
        if self._coordinate_transformation_required:
            self._transformation_stage.tune(env, env_id)
        self._emf_feedforward.tune(env, env_id)
        self._current_base_controller.tune(env, env_id, a)
        self._anti_windup_stage.tune(env, env_id)
        self._clipping_stage.tune(env, env_id)
        self._voltage_reference = np.zeros(
            len(gc.parameter_reader.voltages[gc.utils.get_motor_type(env_id)]), dtype=float
        )
        self._tau_current_loop = gc.parameter_reader.tau_current_loop_reader[motor_type](env)

    def current_control(self, state, current_reference):
        """
        Calculate the input voltages.

        Args:
            state(np.array): state of the environment
            current_reference(np.array): current references

        Returns:
            np.array: voltage references
        """

        # Calculate the voltage reference by the base controllers
        voltage_reference = self._current_base_controller(state, current_reference)

        # Decouple the voltage components
        if self._decoupling:
            voltage_reference = self._emf_feedforward(state, voltage_reference)

        # Clip the voltage inputs to the action space
        self._voltage_reference = self._clipping_stage(state, voltage_reference)

        # Transform the voltages in the correct coordinate system
        if self._coordinate_transformation_required:
            voltage_reference = self._transformation_stage(state, voltage_reference)

        # Integrate the I-Controllers
        if hasattr(self._current_base_controller, "integrator"):
            delta = self._anti_windup_stage(state, current_reference, self._clipping_stage.clipping_difference)
            self._current_base_controller.integrator += delta

        return voltage_reference

    def control(self, state, reference):
        """
        Claculate the reference values for the input voltages.

        Args:
            state(np.array): actual state of the environment
            reference(np.array): current references

        Returns:
            np.ndarray: voltage references
        """

        self._voltage_reference = self.current_control(state, reference)
        return self._voltage_reference

    def reset(self):
        """Reset all components of the stage"""
        for stage in self.stages:
            stage.reset()
