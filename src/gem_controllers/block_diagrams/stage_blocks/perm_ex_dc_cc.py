from control_block_diagram.components import Box, Connection
from control_block_diagram.predefined_components import Add, PIController


def perm_ex_dc_cc(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the Perm Ex DC current control block
    """

    def cc_perm_ex_dc(start, control_task):
        """
        Function to build the Perm Ex DC current control block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # space to the previous block
        space = 1 if control_task == "CC" else 1.5

        # Add block for the current reference and state
        add_current = Add(start.add_x(space))

        if control_task == "CC":
            # Connection at the input
            Connection.connect(
                start, add_current.input_left[0], text=r"$i^{*}$", text_align="left", text_position="start"
            )

        # PI Current Controller
        pi_current = PIController(add_current.position.add_x(1.5), text="Current\nController")

        # Connection between the add block and pi controller
        Connection.connect(add_current.output_right, pi_current.input_left)

        start = pi_current.position  # starting point of the next block
        # Inputs of the stage
        inputs = dict(
            i_ref=[add_current.input_left[0], dict(text=r"$i^{*}$")],
            i=[
                add_current.input_bottom[0],
                dict(text=r"-", move_text=(-0.2, -0.2), text_position="end", text_align="right"),
            ],
        )
        outputs = dict(u=pi_current.output_right[0])  # Outputs of the stage
        connect_to_lines = dict()  # Connections to other lines
        connections = dict()  # Connections

        if emf_feedforward:
            # Add block of the emf feedforward
            add_emf = Add(pi_current.position.add_x(2))

            # Connection between pi controller and add block
            Connection.connect(pi_current.output_right, add_emf.input_left, text=r"$\Delta u^{*}$")

            # Multiplication with the flux
            box_psi = Box(
                add_emf.position.sub_y(2.5),
                size=(0.8, 0.8),
                text=r"$\Psi'_{\mathrm{e}}$",
                inputs=dict(bottom=1),
                outputs=dict(top=1),
            )

            # Connection between the multiplication and add block
            Connection.connect(
                box_psi.output_top,
                add_emf.input_bottom,
                text=r"$u^{0}$",
                text_position="end",
                text_align="right",
                move_text=(-0.1, -0.2),
            )

            # Set the input of the emf feedforward
            if control_task in ["SC"]:
                connect_to_lines["omega"] = [box_psi.input_bottom[0], dict(section=0)]
            elif control_task in ["CC", "TC"]:
                inputs["omega"] = [box_psi.input_bottom[0], dict()]

            # Set the output of the emf feedforward
            outputs["u"] = add_emf.output_right[0]

            # Update the position of the next block
            start = add_emf.position

        return start, inputs, outputs, connect_to_lines, connections

    return cc_perm_ex_dc
