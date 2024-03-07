from control_block_diagram.components import Box, Connection, Point
from control_block_diagram.predefined_components import Add, Limit, PIController


def ext_ex_dc_cc(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the Externally Excited DC current control block
    """

    def cc_ext_ex_dc(start, control_task):
        """
        Function to build the Externally Excited DC current control block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # space to the previous block
        space = 1

        # Add block for the e-current reference and state
        add_i_e = Add(start.add_x(space + 0.5))

        # starting point of the i_a path
        start_i_a = start.sub_y(1.2)

        # Add block for the e-current reference and state
        add_i_a = Add(start_i_a.add_x(space))

        if control_task == "CC":
            # Connection at the input of the i_e path
            Connection.connect(
                start, add_i_e.input_left[0], text=r"$i^{*}_{\mathrm{e}}$", text_align="left", text_position="start"
            )
            # Connection at the input of the i_a path
            Connection.connect(
                start_i_a, add_i_a.input_left[0], text=r"$i^{*}_{\mathrm{a}}$", text_align="left", text_position="start"
            )

        # PI Current Controller of the i_e path
        pi_i_e = PIController(add_i_e.position.add_x(1.5), text="Current\nController")

        # Connection between the PI Controller and the add block
        Connection.connect(add_i_e.output_right, pi_i_e.input_left)

        # PI Current Controller of the i_a path
        pi_i_a = PIController(add_i_a.position.add_x(2))

        # Connection between the PI Controller and the add block
        Connection.connect(add_i_a.output_right, pi_i_a.input_left)

        # Inputs of the stage
        inputs = dict(
            i_e_ref=[add_i_e.input_left[0], dict(text=r"$i^{*}_{\mathrm{e}}$", move_text=(-0.1, 0.1))],
            i_a_ref=[add_i_a.input_left[0], dict(text=r"$i^{*}_{\mathrm{a}}$", move_text=(-0.1, 0.1))],
            i_a=[
                add_i_a.input_bottom[0],
                dict(text=r"-", move_text=(-0.2, -0.2), text_position="end", text_align="right"),
            ],
            i_e=[
                add_i_e.input_bottom[0],
                dict(text=r"-", move_text=(-0.2, -0.2), text_position="end", text_align="right"),
            ],
        )
        connect_to_lines = dict()  # Connections to other lines

        if emf_feedforward:
            # Add block of the emf feedforward
            add_psi = Add(pi_i_a.position.add_x(2))

            # Connection between the PI Controller and the add block
            Connection.connect(
                pi_i_a.output_right, add_psi.input_left, text=r"$\Delta u_{\mathrm{a}}$", distance_y=0.25
            )

            # Multiplication with the flux
            box_psi = Box(
                add_psi.position.sub_y(2),
                size=(1, 0.8),
                text=r"$L'_{\mathrm{e}} i_{\mathrm{e}}$",
                inputs=dict(bottom=1),
                outputs=dict(top=1),
            )

            # Connection between the multiplication and the add block
            Connection.connect(
                box_psi.output_top,
                add_psi.input_bottom,
                text=r"$u_0$",
                text_position="end",
                text_align="right",
                move_text=(-0.05, -0.25),
            )

            # Limit of the i_a current
            limit_a = Limit(add_psi.position.add_x(1.5), size=(1, 1))

            # Connection between the add and limit block
            Connection.connect(add_psi.output_right, limit_a.input_left)

            # Pulse width modulation block of the i_a path
            pwm_a = Box(limit_a.output_right[0].add_x(1.5), size=(1.2, 0.8), text="PWM")

            # Connection between the a-limit and the a-pwm
            Connection.connect(limit_a.output_right, pwm_a.input_left, text=r"$u^{*}_{\mathrm{a}}$", distance_y=0.25)

            # Set the input of the emf feedforward
            if control_task in ["CC", "TC"]:
                inputs["omega"] = [box_psi.input_bottom[0], dict()]
            elif control_task == "SC":
                connect_to_lines["omega"] = [box_psi.input_bottom[0], dict(section=0)]
        else:
            # Limit of the i_a current
            limit_a = Limit(pi_i_a.output_right[0].add_x(1), size=(1, 1))

            # Connection between the add and limit block
            Connection.connect(pi_i_a.output_right, limit_a.input_left)

            # Pulse width modulation block of the i_a path
            pwm_a = Box(limit_a.position.add_x(2), size=(1.2, 0.8), text="PWM")

            # Connection between the a-limit and the a-pwm
            Connection.connect(limit_a.output_right, pwm_a.input_left, text=r"$u^{*}_{\mathrm{a}}$", distance_y=0.25)

        # Limit of the i_a current
        limit_e = Limit(Point.merge(limit_a.position, pi_i_e.position), size=(1, 1))

        # Connection between the PI Controller and the limtit block
        Connection.connect(pi_i_e.output_right, limit_e.input_left)

        # Pulse width modulation block of the i_e path
        pwm_e = Box(Point.merge(pwm_a.position, pi_i_e.position), size=(1.2, 0.8), text="PWM")

        # Connetion between the e-limit and e-pwm block
        Connection.connect(limit_e.output_right, pwm_e.input_left, text=r"$u^{*}_{\mathrm{e}}$", distance_y=0.25)

        start = pwm_e.position  # starting point of the next block
        outputs = dict(u_e=pwm_e.output_right[0], u_a=pwm_a.output_right[0])  # Outputs of the stage
        connections = dict()  # Connections

        return start, inputs, outputs, connect_to_lines, connections

    return cc_ext_ex_dc
