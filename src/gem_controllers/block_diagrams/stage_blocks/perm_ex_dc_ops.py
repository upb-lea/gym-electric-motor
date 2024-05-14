from control_block_diagram.components import Box, Connection
from control_block_diagram.predefined_components import Limit


def perm_ex_dc_ops(start, control_task):
    """
    Function to build the Perm Ex DC operation point selection block
    Args:
        start:          Starting point of the block
        control_task:   Control task of the controller

    Returns:
        endpoint, inputs, outputs, connection to other lines, connections
    """

    # space to the previous block
    space = 1 if control_task == "TC" else 2.2

    # Calculation of the current reference
    box_torque = Box(start.add_x(space), size=(0.8, 0.8), text=r"$\frac{1}{\Psi'_{\mathrm{e}}}$")

    # Limit of the current reference
    limit = Limit(box_torque.output_right[0].add_x(1), size=(1, 1))

    # Connection between the calculation and limit block
    Connection.connect(box_torque.output_right, limit.input_left)

    if control_task == "TC":
        # Connection at the input
        Connection.connect(start, box_torque.input_left[0], text=r"$T^{*}$", text_align="left", text_position="start")

    start = limit.position  # starting point of the next block
    inputs = dict(t_ref=[box_torque.input_left[0], dict(text=r"$T^{*}$")])  # Inputs of the stage
    outputs = dict(i_ref=limit.output_right[0])  # Outputs of the stage
    connect_to_lines = dict()  # Connections to other lines
    connections = dict()  # Connections

    return start, inputs, outputs, connect_to_lines, connections
