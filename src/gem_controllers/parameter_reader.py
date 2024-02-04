import numpy as np

from gym_electric_motor.physical_systems import converters as cv

dc_motors = ["SeriesDc", "ShuntDc", "PermExDc", "ExtExDc"]
synchronous_motors = ["PMSM", "SynRM", "EESM"]
induction_motors = ["DFIM", "SCIM"]
ac_motors = synchronous_motors + induction_motors

control_tasks_ = ["CC", "TC", "SC"]
dc_actions = ["Finite", "Cont"]
ac_actions = ["Finite", "DqCont", "AbcCont"]


psi_reader = {
    "SeriesDc": lambda env: np.array([0.0]),
    "ShuntDc": lambda env: np.array([0.0]),
    "PermExDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["psi_e"]]),
    "ExtExDc": lambda env: np.array([0.0, 0.0]),
    "PMSM": lambda env: np.array([0.0, env.physical_system.electrical_motor.motor_parameter["psi_p"]]),
    "SynRM": lambda env: np.array([0.0, 0.0]),
    "SCIM": lambda env: np.array([0.0, 0.0]),
    "EESM": lambda env: np.array([0.0, 0.0, 0.0]),
}

p_reader = {
    "SeriesDc": lambda env: 1,
    "ShuntDc": lambda env: 1,
    "ExtExDc": lambda env: 0,
    "PermExDc": lambda env: 0,
    "PMSM": lambda env: env.physical_system.electrical_motor.motor_parameter["p"],
    "SynRM": lambda env: env.physical_system.electrical_motor.motor_parameter["p"],
    "SCIM": lambda env: env.physical_system.electrical_motor.motor_parameter["p"],
    "EESM": lambda env: env.physical_system.electrical_motor.motor_parameter["p"],
}

l_reader = {
    "SeriesDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"]
            + env.physical_system.electrical_motor.motor_parameter["l_e"]
        ]
    ),
    "ShuntDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"],
        ]
    ),
    "ExtExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"],
            env.physical_system.electrical_motor.motor_parameter["l_e"],
        ]
    ),
    "PermExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"],
        ]
    ),
    "PMSM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["l_q"],
        ]
    ),
    "SynRM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["l_q"],
        ]
    ),
    "SCIM": lambda env: np.array(
        [
            (
                env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_m"]
            )
            / env.physical_system.electrical_motor.motor_parameter["r_r"],
            (
                env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_m"]
            )
            / env.physical_system.electrical_motor.motor_parameter["r_r"],
        ]
    ),
    "EESM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["l_q"],
            env.physical_system.electrical_motor.motor_parameter["l_e"],
        ]
    ),
}

l_emf_reader = {
    "SeriesDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["l_e_prime"]]),
    "ShuntDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_e_prime"],
        ]
    ),
    "ExtExDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["l_e_prime"], 0.0]),
    "PermExDc": lambda env: np.array([0.0]),
    "PMSM": lambda env: np.array(
        [
            -env.physical_system.electrical_motor.motor_parameter["l_q"],
            env.physical_system.electrical_motor.motor_parameter["l_d"],
        ]
    ),
    "SynRM": lambda env: np.array(
        [
            -env.physical_system.electrical_motor.motor_parameter["l_q"],
            env.physical_system.electrical_motor.motor_parameter["l_d"],
        ]
    ),
    "SCIM": lambda env: np.array(
        [
            -(
                env.physical_system.electrical_motor.motor_parameter["l_sigs"]
                * env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_sigs"]
                * env.physical_system.electrical_motor.motor_parameter["l_m"]
                + env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                * env.physical_system.electrical_motor.motor_parameter["l_m"]
            )
            / (
                env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_m"]
            ),
            (
                env.physical_system.electrical_motor.motor_parameter["l_sigs"]
                * env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_sigs"]
                * env.physical_system.electrical_motor.motor_parameter["l_m"]
                + env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                * env.physical_system.electrical_motor.motor_parameter["l_m"]
            )
            / (
                env.physical_system.electrical_motor.motor_parameter["l_sigr"]
                + env.physical_system.electrical_motor.motor_parameter["l_m"]
            ),
        ]
    ),
    "EESM": lambda env: np.array(
        [
            -env.physical_system.electrical_motor.motor_parameter["l_q"],
            env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["l_m"]
            * env.physical_system.electrical_motor.motor_parameter["l_q"]
            / env.physical_system.electrical_motor.motor_parameter["l_d"],
        ]
    ),
}

tau_current_loop_reader = {
    "SeriesDc": lambda env: np.array(
        [
            (
                env.physical_system.electrical_motor.motor_parameter["l_e"]
                + env.physical_system.electrical_motor.motor_parameter["l_a"]
            )
            / (
                env.physical_system.electrical_motor.motor_parameter["r_e"]
                + env.physical_system.electrical_motor.motor_parameter["r_a"]
            )
        ]
    ),
    "ShuntDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"]
            / env.physical_system.electrical_motor.motor_parameter["r_a"]
        ]
    ),
    "ExtExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_a"]
            / env.physical_system.electrical_motor.motor_parameter["r_a"],
            env.physical_system.electrical_motor.motor_parameter["l_e"]
            / env.physical_system.electrical_motor.motor_parameter["r_e"],
        ]
    ),
    "PermExDc": lambda env: np.array(
        env.physical_system.electrical_motor.motor_parameter["l_a"]
        / env.physical_system.electrical_motor.motor_parameter["r_a"]
    ),
    "PMSM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_q"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["l_d"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
        ]
    ),
    "SynRM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_q"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["l_d"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
        ]
    ),
    "SCIM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_sigs"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["l_sigr"]
            / env.physical_system.electrical_motor.motor_parameter["r_r"],
        ]
    ),
    "EESM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["l_q"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["l_d"]
            / env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["l_e"]
            / env.physical_system.electrical_motor.motor_parameter["r_e"],
        ]
    ),
}

r_reader = {
    "SeriesDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"]
            + env.physical_system.electrical_motor.motor_parameter["r_e"]
        ]
    ),
    "ShuntDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"],
        ]
    ),
    "ExtExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"],
            env.physical_system.electrical_motor.motor_parameter["r_e"],
        ]
    ),
    "PermExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"],
        ]
    ),
    "PMSM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["r_s"],
        ]
    ),
    "SynRM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["r_s"],
        ]
    ),
    "SCIM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["r_r"],
        ]
    ),
    "EESM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["r_s"],
            env.physical_system.electrical_motor.motor_parameter["r_e"],
        ]
    ),
}

tau_n_reader = {
    "SeriesDc": lambda env: np.array(
        [
            (
                env.physical_system.electrical_motor.motor_parameter["r_a"]
                + env.physical_system.electrical_motor.motor_parameter["r_e"]
            )
            / (
                env.physical_system.electrical_motor.motor_parameter["l_a"]
                + env.physical_system.electrical_motor.motor_parameter["l_e"]
            )
        ]
    ),
    "ShuntDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"]
            / env.physical_system.electrical_motor.motor_parameter["l_a"],
        ]
    ),
    "ExtExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"]
            / env.physical_system.electrical_motor.motor_parameter["l_a"],
            env.physical_system.electrical_motor.motor_parameter["r_e"]
            / env.physical_system.electrical_motor.motor_parameter["l_e"],
        ]
    ),
    "PermExDc": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_a"]
            / env.physical_system.electrical_motor.motor_parameter["l_a"],
        ]
    ),
    "PMSM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_q"],
        ]
    ),
    "SynRM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_q"],
        ]
    ),
    "SCIM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_sigs"],
            env.physical_system.electrical_motor.motor_parameter["r_r"]
            / env.physical_system.electrical_motor.motor_parameter["l_sigr"],
        ]
    ),
    "EESM": lambda env: np.array(
        [
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_d"],
            env.physical_system.electrical_motor.motor_parameter["r_s"]
            / env.physical_system.electrical_motor.motor_parameter["l_q"],
            env.physical_system.electrical_motor.motor_parameter["r_e"]
            / env.physical_system.electrical_motor.motor_parameter["l_e"],
        ]
    ),
}

currents = {
    "SeriesDc": ["i"],
    "ShuntDc": ["i_a"],
    "ExtExDc": ["i_a", "i_e"],
    "PermExDc": ["i"],
    "PMSM": ["i_sd", "i_sq"],
    "SynRM": ["i_sd", "i_sq"],
    "SCIM": ["i_sd", "i_sq"],
    "EESM": ["i_sd", "i_sq", "i_e"],
}
emf_currents = {
    "SeriesDc": ["i"],
    "ShuntDc": ["i_e"],
    "ExtExDc": ["i_e", "i_a"],
    "PermExDc": ["i"],
    "PMSM": ["i_sq", "i_sd"],
    "SynRM": ["i_sq", "i_sd"],
    "SCIM": ["i_sq", "i_sd"],
    "EESM": ["i_sq", "i_sd", "i_sq"],
}


voltages = {
    "SeriesDc": ["u"],
    "ShuntDc": ["u"],
    "ExtExDc": ["u_a", "u_e"],
    "PermExDc": ["u"],
    "PMSM": ["u_sd", "u_sq"],
    "SynRM": ["u_sd", "u_sq"],
    "SCIM": ["u_sd", "u_sq"],
    "EESM": ["u_sd", "u_sq", "u_e"],
}


def get_output_voltages(motor_type, action_type):
    if motor_type in dc_motors:
        return voltages[motor_type]
    elif motor_type in dc_motors:
        return voltages[motor_type]
    elif motor_type in dc_motors:
        return voltages[motor_type]
    elif motor_type in induction_motors:
        return ["u_sa", "u_sb", "u_sc"]
    elif motor_type == "EESM":
        return ["u_a", "u_b", "u_c", "u_sup"]
    else:
        return ["u_a", "u_b", "u_c"]


l_prime_reader = {
    "SeriesDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["l_e_prime"]]),
    "ShuntDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["l_e_prime"]]),
    "ExtExDc": lambda env: np.array([env.physical_system.electrical_motor.motor_parameter["l_e_prime"]]),
    "PermExDc": lambda env: np.array([0.0]),
    "PMSM": lambda env: np.array([0.0, 0.0]),
    "SynRM": lambda env: np.array(
        [
            -env.physical_system.electrical_motor.motor_parameter["l_sq"],
            env.physical_system.electrical_motor.motor_parameter["l_sd"],
        ]
    ),
    "SCIM": lambda env: np.array([0, 0]),
    "EESM": lambda env: np.array([0.0, 0.0, 0.0]),
}

converter_high_idle_low_action = {
    cv.FiniteFourQuadrantConverter: (1, 0, 2),
    cv.FiniteTwoQuadrantConverter: (1, 0, 2),
    cv.FiniteOneQuadrantConverter: (1, 0, 0),
    cv.FiniteB6BridgeConverter: ((1, 0, 2),) * 3,
}


class MotorSpecification:
    _motors = dict()

    @staticmethod
    def register(motor_key):
        def reg(motor_specification):
            assert isinstance(motor_specification, MotorSpecification)
            assert motor_key not in MotorSpecification._motors.keys()
            MotorSpecification._motors[motor_key] = motor_specification

        return reg

    @staticmethod
    def get(motor_key):
        return MotorSpecification._motors[motor_key]

    psi = None
    l = None  # noqa: E741
    l_emf = None
    tau_current_loop = None
    tau_n = None
    r = None
    l_prime = None
    currents = None
    emf_currents = None
    voltages = None
