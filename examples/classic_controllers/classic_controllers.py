import numpy as np
from controllers.cascaded_controller import CascadedController
from controllers.cascaded_foc_controller import CascadedFieldOrientedController
from controllers.continuous_action_controller import ContinuousActionController
from controllers.continuous_controller import ContinuousController
from controllers.dicrete_action_controller import DiscreteActionController
from controllers.discrete_controller import DiscreteController
from controllers.foc_controller import FieldOrientedController
from controllers.induction_motor_cascaded_foc import (
    InductionMotorCascadedFieldOrientedController,
)
from controllers.induction_motor_foc import InductionMotorFieldOrientedController
from controllers.on_off_controller import OnOffController
from controllers.pi_controller import PIController
from controllers.pid_controller import PIDController
from controllers.three_point_controller import ThreePointController
from external_plot import ExternalPlot
from externally_referenced_state_plot import ExternallyReferencedStatePlot
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from gym_electric_motor import envs
from gym_electric_motor.physical_systems import (
    DcExternallyExcitedMotor,
    DcMotorSystem,
    DcSeriesMotor,
    DoublyFedInductionMotorSystem,
    SquirrelCageInductionMotorSystem,
    SynchronousMotorSystem,
)
from gym_electric_motor.reference_generators import (
    MultipleReferenceGenerator,
    SwitchedReferenceGenerator,
)
from gym_electric_motor.visualization import MotorDashboard


class Controller:
    """This is the base class for every controller along with the motor environments."""

    @classmethod
    def make(cls, environment, stages=None, **controller_kwargs):
        """
        This function creates the controller structure and optionally tunes the controller.

        Args:
            environment: gym-electric-motor environment to be controlled
            stages: stages of the controller, if no stages are passed, the controller is automatically designed und tuned
            **controller_kwargs: setting parameters for the controller and visualization

        Returns:
            fully designed controller for the control of the gym-electric-motor environment, which is called using the
            control function
            the inputs of the control function are the state and the reference, both are given by the environment

        """

        _controllers = {
            "pi_controller": [
                ContinuousActionController,
                ContinuousController,
                PIController,
            ],
            "pid_controller": [
                ContinuousActionController,
                ContinuousController,
                PIDController,
            ],
            "on_off": [DiscreteActionController, DiscreteController, OnOffController],
            "three_point": [
                DiscreteActionController,
                DiscreteController,
                ThreePointController,
            ],
            "cascaded_controller": [CascadedController],
            "foc_controller": [FieldOrientedController],
            "cascaded_foc_controller": [CascadedFieldOrientedController],
            "foc_rotor_flux_observer": [InductionMotorFieldOrientedController],
            "cascaded_foc_rotor_flux_observer": [InductionMotorCascadedFieldOrientedController],
        }
        controller_kwargs = cls.reference_states(environment, **controller_kwargs)
        controller_kwargs = cls.get_visualization(environment, **controller_kwargs)

        if stages is not None:
            controller_type, stages = cls.find_controller_type(environment, stages, **controller_kwargs)
            assert controller_type in _controllers.keys(), f"Controller {controller_type} unknown"
            stages = cls.automated_gain(environment, stages, controller_type, _controllers, **controller_kwargs)
            controller = _controllers[controller_type][0](environment, stages, _controllers, **controller_kwargs)

        else:
            controller_type, stages = cls.automated_controller_design(environment, **controller_kwargs)
            stages = cls.automated_gain(environment, stages, controller_type, _controllers, **controller_kwargs)
            controller = _controllers[controller_type][0](environment, stages, _controllers, **controller_kwargs)
        return controller

    @staticmethod
    def get_visualization(environment, **controller_kwargs):
        """This method separates external_plots and external_ref_plots. It also checks if a MotorDashboard is used."""
        if "external_plot" in controller_kwargs.keys():
            ext_plot = []
            ref_plot = []
            for external_plots in controller_kwargs["external_plot"]:
                if isinstance(external_plots, ExternalPlot):
                    ext_plot.append(external_plots)
                elif isinstance(external_plots, ExternallyReferencedStatePlot):
                    ref_plot.append(external_plots)
            controller_kwargs["external_plot"] = ext_plot
            controller_kwargs["external_ref_plots"] = ref_plot

        for visualization in environment.unwrapped.visualizations:
            if isinstance(visualization, MotorDashboard):
                controller_kwargs["update_interval"] = visualization.update_interval
                controller_kwargs["visualization"] = True
                return controller_kwargs
        controller_kwargs["visualization"] = False
        return controller_kwargs

    @staticmethod
    def reference_states(environment, **controller_kwargs):
        """This method searches the environment for all referenced states and writes them to an array."""
        ref_states = []
        if isinstance(environment.unwrapped.reference_generator, MultipleReferenceGenerator):
            for rg in environment.unwrapped.reference_generator._sub_generators:
                if isinstance(rg, SwitchedReferenceGenerator):
                    ref_states.append(rg._sub_generators[0]._reference_state)
                else:
                    ref_states.append(rg._reference_state)

        elif isinstance(environment.unwrapped.reference_generator, SwitchedReferenceGenerator):
            ref_states.append(environment.unwrapped.reference_generator._sub_generators[0]._reference_state)
        else:
            ref_states.append(environment.unwrapped.reference_generator._reference_state)
        controller_kwargs["ref_states"] = np.array(ref_states)
        return controller_kwargs

    @staticmethod
    def find_controller_type(environment, stages, **controller_kwargs):
        _stages = stages

        if isinstance(environment.unwrapped.physical_system, DcMotorSystem):
            if type(stages) is list:
                if len(stages) > 1:
                    if type(stages[0]) is list:
                        stages = stages[0]
                    if len(stages) > 1:
                        controller_type = "cascaded_controller"
                    else:
                        controller_type = stages[0]["controller_type"]
                else:
                    controller_type = stages[0]["controller_type"]
            else:
                if type(stages) is dict:
                    controller_type = stages["controller_type"]
                    _stages = [stages]
                else:
                    controller_type = stages
                    _stages = [{"controller_type": stages}]
        elif isinstance(environment.physical_system.unwrapped, SynchronousMotorSystem):
            if len(stages) == 2:
                if len(stages[1]) == 1 and "i_sq" in controller_kwargs["ref_states"]:
                    controller_type = "foc_controller"
                else:
                    controller_type = "cascaded_foc_controller"
            else:
                controller_type = "cascaded_foc_controller"

        elif isinstance(environment.physical_system.unwrapped, SquirrelCageInductionMotorSystem):
            if len(stages) == 2:
                if len(stages[1]) == 1 and "i_sq" in controller_kwargs["ref_states"]:
                    controller_type = "foc_rotor_flux_observer"
                else:
                    controller_type = "cascaded_foc_rotor_flux_observer"
            else:
                controller_type = "cascaded_foc_rotor_flux_observer"

        elif isinstance(environment.physical_system.unwrapped, DoublyFedInductionMotorSystem):
            if len(stages) == 2:
                if len(stages[1]) == 1 and "i_sq" in controller_kwargs["ref_states"]:
                    controller_type = "foc_rotor_flux_observer"
                else:
                    controller_type = "cascaded_foc_rotor_flux_observer"
            else:
                controller_type = "cascaded_foc_rotor_flux_observer"

        return controller_type, _stages

    @staticmethod
    def automated_controller_design(environment, **controller_kwargs):
        """This method automatically designs the controller based on the given motor environment and control task."""

        action_space_type = type(environment.action_space)
        ref_states = controller_kwargs["ref_states"]
        stages = []
        if isinstance(environment.unwrapped.physical_system.unwrapped, DcMotorSystem):  # Checking type of motor
            if "omega" in ref_states or "torque" in ref_states:  # Checking control task
                controller_type = "cascaded_controller"

                for i in range(len(stages), 2):
                    if i == 0:
                        if action_space_type is Box:  # Checking type of output stage (finite / cont)
                            stages.append({"controller_type": "pi_controller"})
                        else:
                            stages.append({"controller_type": "three_point"})
                    else:
                        stages.append({"controller_type": "pi_controller"})  # Adding PI-Controller for overlaid stages

            elif "i" in ref_states or "i_a" in ref_states:
                # Checking type of output stage (finite / cont)
                if action_space_type is Discrete or action_space_type is MultiDiscrete:
                    stages.append({"controller_type": "three_point"})
                elif action_space_type is Box:
                    stages.append({"controller_type": "pi_controller"})
                controller_type = stages[0]["controller_type"]

            # Add stage for i_e current of the ExtExDC
            if isinstance(environment.unwrapped.physical_system.electrical_motor, DcExternallyExcitedMotor):
                if action_space_type is Box:
                    stages = [stages, [{"controller_type": "pi_controller"}]]
                else:
                    stages = [stages, [{"controller_type": "three_point"}]]

        elif isinstance(environment.unwrapped.physical_system.unwrapped, SynchronousMotorSystem):
            if "i_sq" in ref_states or "torque" in ref_states:  # Checking control task
                controller_type = "foc_controller" if "i_sq" in ref_states else "cascaded_foc_controller"
                if action_space_type is Discrete:
                    stages = [
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                    ]
                else:
                    stages = [
                        [
                            {"controller_type": "pi_controller"},
                            {"controller_type": "pi_controller"},
                        ]
                    ]
            elif "omega" in ref_states:
                controller_type = "cascaded_foc_controller"
                if action_space_type is Discrete:
                    stages = [
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "pi_controller"}],
                    ]
                else:
                    stages = [
                        [
                            {"controller_type": "pi_controller"},
                            {"controller_type": "pi_controller"},
                        ],
                        [{"controller_type": "pi_controller"}],
                    ]

        elif isinstance(
            environment.physical_system.unwrapped,
            (SquirrelCageInductionMotorSystem, DoublyFedInductionMotorSystem),
        ):
            if "i_sq" in ref_states or "torque" in ref_states:
                controller_type = (
                    "foc_rotor_flux_observer" if "i_sq" in ref_states else "cascaded_foc_rotor_flux_observer"
                )
                if action_space_type is Discrete:
                    stages = [
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                    ]
                else:
                    stages = [
                        [
                            {"controller_type": "pi_controller"},
                            {"controller_type": "pi_controller"},
                        ]
                    ]
            elif "omega" in ref_states:
                controller_type = "cascaded_foc_rotor_flux_observer"
                if action_space_type is Discrete:
                    stages = [
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "on_off"}],
                        [{"controller_type": "pi_controller"}],
                    ]
                else:
                    stages = [
                        [
                            {"controller_type": "pi_controller"},
                            {"controller_type": "pi_controller"},
                        ],
                        [{"controller_type": "pi_controller"}],
                    ]
        else:
            controller_type = "foc_controller"

        return controller_type, stages

    @staticmethod
    def automated_gain(environment, stages, controller_type, _controllers, **controller_kwargs):
        """
        This method automatically parameterizes a given controller design if the parameter automated_gain is True
        (default True), based on the design according to the symmetric optimum (SO). Further information about the
        design according to the SO can be found in the following paper (https://ieeexplore.ieee.org/document/55967).

        Args:
            environment: gym-electric-motor environment
            stages: list of the stages of the controller
            controller_type: string of the used controller type from the dictionary _controllers
            _controllers: dictionary of all possible controllers and controller stages
            controller_kwargs: further arguments of the controller

        Returns:
            list of stages, which are completely parameterized
        """

        ref_states = controller_kwargs["ref_states"]
        mp = environment.unwrapped.physical_system.electrical_motor.motor_parameter
        limits = environment.unwrapped.physical_system.limits
        omega_lim = limits[environment.unwrapped.state_names.index("omega")]
        if isinstance(environment.unwrapped.physical_system.unwrapped, DcMotorSystem):
            i_a_lim = limits[environment.unwrapped.physical_system.CURRENTS_IDX[0]]
            i_e_lim = limits[environment.unwrapped.physical_system.CURRENTS_IDX[-1]]
            u_a_lim = limits[environment.unwrapped.physical_system.VOLTAGES_IDX[0]]
            u_e_lim = limits[environment.unwrapped.physical_system.VOLTAGES_IDX[-1]]

        elif isinstance(environment.physical_system.unwrapped, SynchronousMotorSystem):
            i_sd_lim = limits[environment.state_names.index("i_sd")]
            i_sq_lim = limits[environment.state_names.index("i_sq")]
            u_sd_lim = limits[environment.state_names.index("u_sd")]
            u_sq_lim = limits[environment.state_names.index("u_sq")]
            torque_lim = limits[environment.state_names.index("torque")]

        else:
            i_sd_lim = limits[environment.state_names.index("i_sd")]
            i_sq_lim = limits[environment.state_names.index("i_sq")]
            u_sd_lim = limits[environment.state_names.index("u_sd")]
            u_sq_lim = limits[environment.state_names.index("u_sq")]
            torque_lim = limits[environment.state_names.index("torque")]

        # The parameter a is a design parameter when designing a controller according to the SO
        a = controller_kwargs.get("a", 4)
        automated_gain = controller_kwargs.get("automated_gain", True)

        if isinstance(environment.unwrapped.physical_system.electrical_motor, DcSeriesMotor):
            mp["l"] = mp["l_a"] + mp["l_e"]
        elif isinstance(environment.unwrapped.physical_system.unwrapped, DcMotorSystem):
            mp["l"] = mp["l_a"]

        if "automated_gain" not in controller_kwargs.keys() or automated_gain:
            cont_extex_envs = (
                envs.ContSpeedControlDcExternallyExcitedMotorEnv,
                envs.ContCurrentControlDcExternallyExcitedMotorEnv,
                envs.ContTorqueControlDcExternallyExcitedMotorEnv,
            )
            finite_extex_envs = (
                envs.FiniteTorqueControlDcExternallyExcitedMotorEnv,
                envs.FiniteSpeedControlDcExternallyExcitedMotorEnv,
                envs.FiniteCurrentControlDcExternallyExcitedMotorEnv,
            )
            if type(environment) in cont_extex_envs:
                stages_a = stages[0]
                stages_e = stages[1]

                p_gain = mp["l_e"] / (environment.physical_system.tau * a) / u_e_lim * i_e_lim
                i_gain = p_gain / (environment.physical_system.tau * a**2)

                stages_e[0]["p_gain"] = stages_e[0].get("p_gain", p_gain)
                stages_e[0]["i_gain"] = stages_e[0].get("i_gain", i_gain)

                if stages_e[0]["controller_type"] == PIDController:
                    d_gain = p_gain * environment.physical_system.tau
                    stages_e[0]["d_gain"] = stages_e[0].get("d_gain", d_gain)
            elif type(environment) in finite_extex_envs:
                stages_a = stages[0]
                stages_e = stages[1]
            else:
                stages_a = stages
                stages_e = False

            if _controllers[controller_type][0] == ContinuousActionController:
                if "i" in ref_states or "i_a" in ref_states or "torque" in ref_states:
                    p_gain = mp["l"] / (environment.physical_system.tau * a) / u_a_lim * i_a_lim
                    i_gain = p_gain / (environment.physical_system.tau * a**2)

                    stages_a[0]["p_gain"] = stages_a[0].get("p_gain", p_gain)
                    stages_a[0]["i_gain"] = stages_a[0].get("i_gain", i_gain)

                    if _controllers[controller_type][2] == PIDController:
                        d_gain = p_gain * environment.physical_system.tau
                        stages_a[0]["d_gain"] = stages_a[0].get("d_gain", d_gain)

                elif "omega" in ref_states:
                    p_gain = (
                        environment.physical_system.mechanical_load.j_total
                        * mp["r_a"] ** 2
                        / (a * mp["l"])
                        / u_a_lim
                        * omega_lim
                    )
                    i_gain = p_gain / (a * mp["l"])

                    stages_a[0]["p_gain"] = stages_a[0].get("p_gain", p_gain)
                    stages_a[0]["i_gain"] = stages_a[0].get("i_gain", i_gain)

                    if _controllers[controller_type][2] == PIDController:
                        d_gain = p_gain * environment.physical_system.tau
                        stages_a[0]["d_gain"] = stages_a[0].get("d_gain", d_gain)

            elif _controllers[controller_type][0] == CascadedController:
                for i in range(len(stages)):
                    if type(stages_a[i]) is list:
                        if (
                            _controllers[stages_a[i][0]["controller_type"]][1] == ContinuousController
                        ):  # had to add [0] to make dict in list acessable
                            if i == 0:
                                p_gain = mp["l"] / (environment.unwrapped.physical_system.tau * a) / u_a_lim * i_a_lim
                                i_gain = p_gain / (environment.unwrapped.physical_system.tau * a**2)

                                if _controllers[stages_a[i][0]["controller_type"]][2] == PIDController:
                                    d_gain = p_gain * environment.unwrapped.physical_system.tau
                                    stages_a[i][0]["d_gain"] = stages_a[i][0].get("d_gain", d_gain)

                            elif i == 1:
                                t_n = environment.unwrapped.physical_system.tau * a**2
                                p_gain = (
                                    environment.unwrapped.physical_system.mechanical_load.j_total
                                    / (a * t_n)
                                    / i_a_lim
                                    * omega_lim
                                )
                                i_gain = p_gain / (a * t_n)
                                if _controllers[stages_a[i][0]["controller_type"]][2] == PIDController:
                                    d_gain = p_gain * environment.unwrapped.physical_system.tau
                                    stages_a[i][0]["d_gain"] = stages_a[i][0].get("d_gain", d_gain)

                            stages_a[i][0]["p_gain"] = stages_a[i][0].get("p_gain", p_gain)  # ?
                            stages_a[i][0]["i_gain"] = stages_a[i][0].get("i_gain", i_gain)  # ?

                    elif type(stages_a[i]) is dict:
                        if (
                            _controllers[stages_a[i]["controller_type"]][1] == ContinuousController
                        ):  # had to add [0] to make dict in list acessable
                            if i == 0:
                                p_gain = mp["l"] / (environment.unwrapped.physical_system.tau * a) / u_a_lim * i_a_lim
                                i_gain = p_gain / (environment.unwrapped.physical_system.tau * a**2)

                                if _controllers[stages_a[i]["controller_type"]][2] == PIDController:
                                    d_gain = p_gain * environment.physical_system.tau
                                    stages_a[i]["d_gain"] = stages_a[i].get("d_gain", d_gain)

                            elif i == 1:
                                t_n = environment.unwrapped.physical_system.tau * a**2
                                p_gain = (
                                    environment.unwrapped.physical_system.mechanical_load.j_total
                                    / (a * t_n)
                                    / i_a_lim
                                    * omega_lim
                                )
                                i_gain = p_gain / (a * t_n)
                                if _controllers[stages_a[i]["controller_type"]][2] == PIDController:
                                    d_gain = p_gain * environment.unwrapped.physical_system.tau
                                    stages_a[i]["d_gain"] = stages_a[i].get("d_gain", d_gain)

                            stages_a[i]["p_gain"] = stages_a[i].get("p_gain", p_gain)  # ?
                            stages_a[i]["i_gain"] = stages_a[i].get("i_gain", i_gain)  # ?

                stages = stages_a if not stages_e else [stages_a, stages_e]

            elif _controllers[controller_type][0] == FieldOrientedController:
                if type(environment.action_space) == Box:
                    stage_d = stages[0][0]
                    stage_q = stages[0][1]
                    if "i_sq" in ref_states and _controllers[stage_q["controller_type"]][1] == ContinuousController:
                        p_gain_d = mp["l_d"] / (1.5 * environment.physical_system.tau * a) / u_sd_lim * i_sd_lim
                        i_gain_d = p_gain_d / (1.5 * environment.physical_system.tau * a**2)

                        p_gain_q = mp["l_q"] / (1.5 * environment.physical_system.tau * a) / u_sq_lim * i_sq_lim
                        i_gain_q = p_gain_q / (1.5 * environment.physical_system.tau * a**2)

                        stage_d["p_gain"] = stage_d.get("p_gain", p_gain_d)
                        stage_d["i_gain"] = stage_d.get("i_gain", i_gain_d)

                        stage_q["p_gain"] = stage_q.get("p_gain", p_gain_q)
                        stage_q["i_gain"] = stage_q.get("i_gain", i_gain_q)

                        if _controllers[stage_d["controller_type"]][2] == PIDController:
                            d_gain_d = p_gain_d * environment.physical_system.tau
                            stage_d["d_gain"] = stage_d.get("d_gain", d_gain_d)

                        if _controllers[stage_q["controller_type"]][2] == PIDController:
                            d_gain_q = p_gain_q * environment.physical_system.tau
                            stage_q["d_gain"] = stage_q.get("d_gain", d_gain_q)
                        stages = [[stage_d, stage_q]]

            elif _controllers[controller_type][0] == CascadedFieldOrientedController:
                if type(environment.action_space) is Box:
                    stage_d = stages[0][0]
                    stage_q = stages[0][1]
                    if "torque" not in controller_kwargs["ref_states"]:
                        overlaid = stages[1]

                    p_gain_d = mp["l_d"] / (1.5 * environment.physical_system.tau * a) / u_sd_lim * i_sd_lim
                    i_gain_d = p_gain_d / (1.5 * environment.physical_system.tau * a**2)

                    p_gain_q = mp["l_q"] / (1.5 * environment.physical_system.tau * a) / u_sq_lim * i_sq_lim
                    i_gain_q = p_gain_q / (1.5 * environment.physical_system.tau * a**2)

                    stage_d["p_gain"] = stage_d.get("p_gain", p_gain_d)
                    stage_d["i_gain"] = stage_d.get("i_gain", i_gain_d)

                    stage_q["p_gain"] = stage_q.get("p_gain", p_gain_q)
                    stage_q["i_gain"] = stage_q.get("i_gain", i_gain_q)

                    if _controllers[stage_d["controller_type"]][2] == PIDController:
                        d_gain_d = p_gain_d * environment.physical_system.tau
                        stage_d["d_gain"] = stage_d.get("d_gain", d_gain_d)

                    if _controllers[stage_q["controller_type"]][2] == PIDController:
                        d_gain_q = p_gain_q * environment.physical_system.tau
                        stage_q["d_gain"] = stage_q.get("d_gain", d_gain_q)

                    if (
                        "torque" not in controller_kwargs["ref_states"]
                        and _controllers[overlaid[0]["controller_type"]][1] == ContinuousController
                    ):
                        t_n = p_gain_d / i_gain_d
                        j_total = environment.physical_system.mechanical_load.j_total
                        p_gain = j_total / (a**2 * t_n) / torque_lim * omega_lim
                        i_gain = p_gain / (a * t_n)

                        overlaid[0]["p_gain"] = overlaid[0].get("p_gain", p_gain)
                        overlaid[0]["i_gain"] = overlaid[0].get("i_gain", i_gain)

                        if _controllers[overlaid[0]["controller_type"]][2] == PIDController:
                            d_gain = p_gain * environment.physical_system.tau
                            overlaid[0]["d_gain"] = overlaid[0].get("d_gain", d_gain)

                        stages = [[stage_d, stage_q], overlaid]

                    else:
                        stages = [[stage_d, stage_q]]

                else:
                    if (
                        "omega" in ref_states
                        and _controllers[stages[3][0]["controller_type"]][1] == ContinuousController
                    ):
                        p_gain = (
                            environment.physical_system.mechanical_load.j_total
                            / (1.5 * a**2 * mp["p"] * np.abs(mp["l_d"] - mp["l_q"]))
                            / i_sq_lim
                            * omega_lim
                        )
                        i_gain = p_gain / (1.5 * environment.physical_system.tau * a)

                        stages[3][0]["p_gain"] = stages[3][0].get("p_gain", p_gain)
                        stages[3][0]["i_gain"] = stages[3][0].get("i_gain", i_gain)

                        if _controllers[stages[3][0]["controller_type"]][2] == PIDController:
                            d_gain = p_gain * environment.physical_system.tau
                            stages[3][0]["d_gain"] = stages[3][0].get("d_gain", d_gain)

            elif _controllers[controller_type][0] == InductionMotorFieldOrientedController:
                mp["l_s"] = mp["l_m"] + mp["l_sigs"]
                mp["l_r"] = mp["l_m"] + mp["l_sigr"]
                sigma = (mp["l_s"] * mp["l_r"] - mp["l_m"] ** 2) / (mp["l_s"] * mp["l_r"])
                tau_sigma = (sigma * mp["l_s"]) / (mp["r_s"] + mp["r_r"] * mp["l_m"] ** 2 / mp["l_r"] ** 2)
                tau_r = mp["l_r"] / mp["r_r"]
                p_gain = tau_r / tau_sigma
                i_gain = p_gain / tau_sigma

                stages[0][0]["p_gain"] = stages[0][0].get("p_gain", p_gain)
                stages[0][0]["i_gain"] = stages[0][0].get("i_gain", i_gain)
                stages[0][1]["p_gain"] = stages[0][1].get("p_gain", p_gain)
                stages[0][1]["i_gain"] = stages[0][1].get("i_gain", i_gain)

                if _controllers[stages[0][0]["controller_type"]][2] == PIDController:
                    d_gain = p_gain * tau_sigma
                    stages[0][0]["d_gain"] = stages[0][0].get("d_gain", d_gain)

                if _controllers[stages[0][1]["controller_type"]][2] == PIDController:
                    d_gain = p_gain * tau_sigma
                    stages[0][1]["d_gain"] = stages[0][1].get("d_gain", d_gain)

            elif _controllers[controller_type][0] == InductionMotorCascadedFieldOrientedController:
                if "torque" not in controller_kwargs["ref_states"]:
                    overlaid = stages[1]

                mp["l_s"] = mp["l_m"] + mp["l_sigs"]
                mp["l_r"] = mp["l_m"] + mp["l_sigr"]
                sigma = (mp["l_s"] * mp["l_r"] - mp["l_m"] ** 2) / (mp["l_s"] * mp["l_r"])
                tau_sigma = (sigma * mp["l_s"]) / (mp["r_s"] + mp["r_r"] * mp["l_m"] ** 2 / mp["l_r"] ** 2)
                tau_r = mp["l_r"] / mp["r_r"]
                p_gain = tau_r / tau_sigma
                i_gain = p_gain / tau_sigma

                stages[0][0]["p_gain"] = stages[0][0].get("p_gain", p_gain)
                stages[0][0]["i_gain"] = stages[0][0].get("i_gain", i_gain)
                stages[0][1]["p_gain"] = stages[0][1].get("p_gain", p_gain)
                stages[0][1]["i_gain"] = stages[0][1].get("i_gain", i_gain)

                if _controllers[stages[0][0]["controller_type"]][2] == PIDController:
                    d_gain = p_gain * tau_sigma
                    stages[0][0]["d_gain"] = stages[0][0].get("d_gain", d_gain)

                if _controllers[stages[0][1]["controller_type"]][2] == PIDController:
                    d_gain = p_gain * tau_sigma
                    stages[0][1]["d_gain"] = stages[0][1].get("d_gain", d_gain)

                if (
                    "torque" not in controller_kwargs["ref_states"]
                    and _controllers[overlaid[0]["controller_type"]][1] == ContinuousController
                ):
                    t_n = p_gain / i_gain
                    j_total = environment.physical_system.mechanical_load.j_total
                    p_gain = j_total / (a**2 * t_n) / torque_lim * omega_lim
                    i_gain = p_gain / (a * t_n)

                    overlaid[0]["p_gain"] = overlaid[0].get("p_gain", p_gain)
                    overlaid[0]["i_gain"] = overlaid[0].get("i_gain", i_gain)

                    if _controllers[overlaid[0]["controller_type"]][2] == PIDController:
                        d_gain = p_gain * environment.physical_system.tau
                        overlaid[0]["d_gain"] = overlaid[0].get("d_gain", d_gain)

                    stages = [stages[0], overlaid]

        return stages
