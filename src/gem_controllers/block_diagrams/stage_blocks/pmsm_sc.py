from control_block_diagram.components import Connection
from control_block_diagram.predefined_components import Add, Limit, PIController


def pmsm_speed_controller(start, control_task):
    """
    Function to build the speed controller block of the PMSM
    Args:
        start:          Starting point of the block
        control_task:   Control task of the controller

    Returns:
        endpoint, inputs, outputs, connection to other lines, connections
    """

    # space to the previous block
    space = 1 if control_task == "SC" else 1.5

    # Add block for the speed reference and state
    add_omega = Add(start.add_x(space))

    # Connection between the mechanical speed input and the add block
    Connection.connect(
        add_omega.input_bottom[0].sub_y(1),
        add_omega.input_bottom[0],
        text=r"$\omega_{\mathrm{me}}$",
        text_align="bottom",
        text_position="start",
    )

    if control_task == "SC":
        # Connection at the input
        Connection.connect(
            start, add_omega.input_left[0], text=r"$\omega^{*}$", text_align="left", text_position="start"
        )

    # PI Speed Controller
    pi_omega = PIController(add_omega.position.add_x(1.5), text="Speed\nController")

    # Connection between the add block and pi controller
    Connection.connect(add_omega.output_right, pi_omega.input_left)

    # Limit of the torque reference
    limit = Limit(pi_omega.output_right[0].add_x(1.5), size=(1, 1))

    # Connection between the pi controller and the limit block
    Connection.connect(pi_omega.output_right[0], limit.input_left[0])

    start = limit.output_right[0]  # starting point of the next block
    inputs = dict(omega_ref=[add_omega.input_left[0], dict(text=r"$\omega^{*}$")])  # Inputs of the stage
    outputs = dict(t_ref=limit.output_right[0])  # Outputs of the stage
    connect_to_lines = dict()  # Connections to other lines
    connections = dict()  # Conncetions of the stage

    return start, inputs, outputs, connect_to_lines, connections
