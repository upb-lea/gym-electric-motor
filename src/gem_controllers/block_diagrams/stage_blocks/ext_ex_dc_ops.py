from control_block_diagram.components import Box, Circle, Connection, Point
from control_block_diagram.predefined_components import Divide, Limit, Multiply


def ext_ex_dc_ops(start, control_task):
    """
    Function to build the Externally Excited DC operation point selection block
    Args:
        start:          Starting point of the block
        control_task:   Control task of the controller

    Returns:
        endpoint, inputs, outputs, connection to other lines, connections
    """

    # space to the previous block
    inp = start.add_x(1) if control_task == "TC" else start.add_x(1.5)

    # Connector at the input
    Circle(inp, radius=0.05, fill="black")

    if control_task == "TC":
        # Connetion at the input
        Connection.connect(start, inp, text=r"$T^{*}$", text_align="left", text_position="start", arrow=False)

    # Calculate absolute value
    box_abs = Box(inp.add(1, 0.6), size=(0.8, 0.8), text=r"|x|")

    # Connect the input and the previous block
    Connection.connect(inp, box_abs.input_left[0], start_direction="north")

    # Multiply by motor parameters
    multiply = Multiply(box_abs.position.add_x(1.5), size=(0.8, 0.8), inputs=dict(left=1, top=1))

    # Connect the previous blocks
    Connection.connect(box_abs.output_right, multiply.input_left)

    # Block of motor parameters
    box_rl = Box(
        multiply.position.add_y(1.2),
        size=(1.5, 0.8),
        text=r"$\sqrt{\frac{R_{\mathrm{a}}}{R_{\mathrm{e}}}}L'_{\mathrm{e}}$",
        outputs=dict(bottom=1),
    )

    # Connect the motor parameters block with the multiply block
    Connection.connect(box_rl.output_bottom, multiply.input_top)

    # Square root of the product
    box_sqrt = Box(multiply.position.add_x(1.5), size=(0.8, 0.8), text=r"$\sqrt{x}$")

    # Connection between multiplication and squareroot
    Connection.connect(multiply.output_right, box_sqrt.input_left)

    # Divide the torque reference by l_e prime
    divide_1 = Divide(multiply.position.sub_y(1.5), size=(0.8, 1.2), input_space=0.6)

    # Conncet the input and division block
    Connection.connect(inp, divide_1.input_left[0], start_direction="south")

    # L_e prime block
    box_le = Box(divide_1.input_left[1].sub_x(1), size=(0.8, 0.8), text=r"$L'_{\mathrm{e}}$")

    # Conncet the l_e prime and division blcok
    Connection.connect(box_le.output_right[0], divide_1.input_left[1])

    # Divide the reference for i_e
    divide_2 = Divide(divide_1.output_right[0].add(3, 0.3), size=(0.8, 1.2), input_space=0.6)

    # Connection between the two division blocks
    Connection.connect(divide_1.output_right[0], divide_2.input_left[1])

    # Limit the i_e current
    limit_i_e = Limit(box_sqrt.output_right[0].add_x(3), size=(1, 1))

    # Connect between the squareroot and limit block
    con_sqrt = Connection.connect(box_sqrt.output_right[0], limit_i_e.input_left[0])

    # Connect the previous line to the second division block
    Connection.connect_to_line(con_sqrt, divide_2.input_left[0].sub_x(0.5), arrow=False)
    Connection.connect(divide_2.input_left[0].sub_x(0.5), divide_2.input_left[0])

    # Limit the i_a current
    limit_i_a = Limit(Point.merge(limit_i_e.position, divide_2.position), size=(1, 1))

    # Connection between the division and limit block
    Connection.connect(divide_2.output_right, limit_i_a.input_left)

    inputs = dict(t_ref=[inp, dict(text=r"$T^{*}$", arrow=False)])  # starting point of the next block
    outputs = dict(i_e_ref=limit_i_e.output_right[0], i_a_ref=limit_i_a.output_right[0])  # Inputs of the stage
    connect_to_lines = dict()  # Outputs of the stage
    connections = dict()  # Connections to other lines
    start = limit_i_e.output_right[0]  # Connections

    return start, inputs, outputs, connect_to_lines, connections
