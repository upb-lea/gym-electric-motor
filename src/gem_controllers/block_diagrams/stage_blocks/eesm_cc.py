from control_block_diagram.components import Box, Circle, Connection, Point
from control_block_diagram.predefined_components import (
    AbcToAlphaBetaTransformation,
    Add,
    AlphaBetaToDqTransformation,
    DqToAbcTransformation,
    Limit,
    Multiply,
    PIController,
)


def eesm_cc(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the EESM current control block
    """

    def cc_eesm(start, control_task):
        """
        Function to build the EESM current control block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # space to the previous block
        space = 1 if control_task == "CC" else 1.5

        # Add blocks for the i_sd and i_sq references and states
        add_i_e = Add(start.add(space - 0.5, 1))
        add_i_sd = Add(start.add_x(space + 0.5))
        add_i_sq = Add(start.add(space, -1))

        if control_task == "CC":
            # Connections at the inputs
            Connection.connect(
                add_i_sd.input_left[0].sub_x(space + 1),
                add_i_sd.input_left[0],
                text=r"$i^{*}_{\mathrm{sd}}$",
                text_position="start",
                text_align="left",
            )
            Connection.connect(
                add_i_sq.input_left[0].sub_x(space + 0.5),
                add_i_sq.input_left[0],
                text=r"$i^{*}_{\mathrm{sq}}$",
                text_position="start",
                text_align="left",
            )
            Connection.connect(
                add_i_e.input_left[0].sub_x(space),
                add_i_e.input_left[0],
                text=r"$i^{*}_{\mathrm{e}}$",
                text_position="start",
                text_align="left",
            )

        # PI Controllers for the d and q component
        pi_i_sd = PIController(add_i_sd.position.add_x(1.2), size=(1, 0.8), input_number=1, output_number=1)
        pi_i_sq = PIController(
            Point.merge(pi_i_sd.position, add_i_sq.position), size=(1, 0.8), input_number=1, output_number=1
        )
        pi_i_e = PIController(
            Point.merge(pi_i_sd.position, add_i_e.position),
            size=(1, 0.8),
            input_number=1,
            output_number=1,
            text="Current\nController",
        )

        # Connection between the add blocks and the PI Controllers
        Connection.connect(add_i_sd.output_right, pi_i_sd.input_left)
        Connection.connect(add_i_sq.output_right, pi_i_sq.input_left)
        Connection.connect(add_i_e.output_right, pi_i_e.input_left)

        # Add blocks for the EMF Feedforward
        add_u_sd = Add(pi_i_sd.position.add_x(1.6))
        add_u_sq = Add(pi_i_sq.position.add_x(1.2))
        add_u_e = Add(pi_i_e.position.add_x(2))

        # Connections between the PI Controllers and the Add blocks
        Connection.connect(
            pi_i_sd.output_right[0], add_u_sd.input_left[0], text=r"$\Delta u^{*}_{\mathrm{sd}}$", distance_y=0.28
        )
        Connection.connect(
            pi_i_sq.output_right[0],
            add_u_sq.input_left[0],
            text=r"$\Delta u^{*}_{\mathrm{sq}}$",
            distance_y=0.4,
            move_text=(0.25, 0),
        )

        # Coordinate transformation from DQ to Abc coordinates
        dq_to_abc = DqToAbcTransformation(Point.get_mid(add_u_sd.position, add_u_sq.position).add_x(2), input_space=1)

        # Connections between the add blocks and the coordinate transformation
        Connection.connect(
            add_u_sd.output_right[0],
            dq_to_abc.input_left[0],
            text=r"$u^{*}_{\mathrm{sd}}$",
            distance_y=0.28,
            move_text=(0.18, 0),
        )
        Connection.connect(
            add_u_sq.output_right[0],
            dq_to_abc.input_left[1],
            text=r"$u^{*}_{\mathrm{sq}}$",
            distance_y=0.28,
            move_text=(0.38, 0),
        )

        # Limit of the input voltages
        limit = Limit(
            dq_to_abc.position.add_x(1.8),
            size=(1, 1.2),
            inputs=dict(left=3, left_space=0.3),
            outputs=dict(right=3, right_space=0.3),
        )

        limit_e = Limit(
            Point.merge(limit.position, pi_i_e.position), size=(1, 1), inputs=dict(left=1), outputs=dict(right=1)
        )

        # Connection between the coordinate transformation and the limit block
        Connection.connect(dq_to_abc.output_right, limit.input_left)
        Connection.connect(pi_i_e.output_right, limit_e.input_left)

        # Pulse width modulation block
        pwm = Box(
            limit.position.add_x(2.5),
            size=(1.5, 1.2),
            text="PWM",
            inputs=dict(left=3, left_space=0.3),
            outputs=dict(right=3, right_space=0.3),
        )

        pwm_e = Box(limit_e.position.add_x(2.5), size=(1.5, 1), text="PWM", inputs=dict(left=1), outputs=dict(right=1))

        # Connection between the limit and the PWM block
        Connection.connect(
            limit.output_right,
            pwm.input_left,
            text=["", "", r"$u^*_{\mathrm{s a,b,c}}$"],
            distance_y=0.25,
            text_align="bottom",
        )
        Connection.connect(limit_e.output_right, pwm_e.input_left, text=[r"$u^*_{\mathrm{e}}$"])

        # Coordinate transformation from ABC to AlphaBeta coordinates
        abc_to_alpha_beta = AbcToAlphaBetaTransformation(pwm.position.sub(1, 3.5), input="right", output="left")

        # Coordinate transformation from AlphaBeta to DQ coordinates
        alpha_beta_to_dq = AlphaBetaToDqTransformation(
            Point.merge(dq_to_abc.position, abc_to_alpha_beta.position), input="right", output="left"
        )

        emf = Box(
            Point.merge(add_u_sd.position, Point.get_mid(dq_to_abc.position, alpha_beta_to_dq.position)),
            size=(2, 1),
            inputs=dict(bottom=4),
            outputs=dict(top=3, top_space=0.4),
            text="EMF Feedforward",
        )

        Connection.connect(emf.output_top[0], add_u_sq.input_bottom[0])
        Connection.connect(emf.output_top[1], add_u_sd.input_bottom[0])
        Connection.connect(emf.output_top[2], add_u_e.input_bottom[0])

        # Connections between the coordinate transformation and the add blocks
        con_3 = Connection.connect(
            alpha_beta_to_dq.output_left[0],
            add_i_sd.input_bottom[0],
            text=r"-",
            text_position="end",
            text_align="right",
            move_text=(-0.2, -0.2),
        )
        con_4 = Connection.connect(
            alpha_beta_to_dq.output_left[1],
            add_i_sq.input_bottom[0],
            text=r"-",
            text_position="end",
            text_align="right",
            move_text=(-0.2, -0.2),
        )

        # Connections between the previous conncetions and the inductances blocks
        Connection.connect_to_line(con_3, emf.input_bottom[3])
        Connection.connect_to_line(con_4, emf.input_bottom[2])

        # Derivation of the angle
        box_d_dt = Box(
            Point.merge(alpha_beta_to_dq.position, start.sub_y(7.42020066645696)).sub_x(1.5),
            size=(1, 0.8),
            text=r"$\mathrm{d} / \mathrm{d}t$",
            inputs=dict(right=1),
            outputs=dict(left=1),
        )

        # Conncetion between the derivation and multiplication block
        con_omega = Connection.connect(
            box_d_dt.output_left[0],
            emf.input_bottom[0],
            space_y=1,
            text=r"$\omega_{\mathrm{el}}$",
            move_text=(0, 1.7),
            text_align="left",
            distance_x=0.3,
        )

        if control_task == "SC":
            # Connector at the previous connection
            Circle(con_omega.points[1], radius=0.05, fill="black")

        # Conncetion between the coordinate transformations
        Connection.connect(
            abc_to_alpha_beta.output,
            alpha_beta_to_dq.input_right,
            text=[r"$i_{\mathrm{s} \upalpha}$", r"$i_{\mathrm{s} \upbeta}$"],
        )

        # Add block for the advanced angle
        add = Add(
            Point.get_mid(dq_to_abc.position, alpha_beta_to_dq.position),
            inputs=dict(bottom=1, right=1),
            outputs=dict(top=1),
        )

        # Connections of the add block
        Connection.connect(
            alpha_beta_to_dq.output_top,
            add.input_bottom,
            text=r"$\varepsilon_{\mathrm{el}}$",
            text_align="right",
            move_text=(0, -0.1),
        )
        Connection.connect(add.output_top, dq_to_abc.input_bottom)

        # Calculate the advanced angle
        box_t_a = Box(
            add.position.add_x(1.5),
            size=(1, 0.8),
            text=r"$1.5 T_{\mathrm{s}}$",
            inputs=dict(right=1),
            outputs=dict(left=1),
        )

        # Connections of the advanced angle block
        Connection.connect(box_t_a.output, add.input_right, text=r"$\Delta \varepsilon$")
        Connection.connect(
            box_t_a.input[0].add_x(0.5),
            box_t_a.input[0],
            text=r"$\omega_{\mathrm{el}}$",
            text_position="start",
            text_align="right",
            distance_x=0.3,
        )

        start = pwm.position  # starting point of the next block

        # Inputs of the stage
        inputs = dict(
            i_d_ref=[
                add_i_sd.input_left[0],
                dict(
                    text=r"$i^{*}_{\mathrm{sd}}$",
                    distance_y=0.25,
                    move_text=(0.3, 0),
                    text_position="start",
                    text_aglin="top",
                ),
            ],
            i_q_ref=[
                add_i_sq.input_left[0],
                dict(
                    text=r"$i^{*}_{\mathrm{sq}}$",
                    distance_y=0.25,
                    move_text=(0.3, 0),
                    text_position="start",
                    text_aglin="top",
                ),
            ],
            i_e_ref=[
                add_i_e.input_left[0],
                dict(
                    text=r"$i^{*}_{\mathrm{e}}$",
                    distance_y=0.25,
                    move_text=(0.3, 0),
                    text_position="start",
                    text_aglin="top",
                ),
            ],
            epsilon=[box_d_dt.input_right[0], dict()],
            i_e=[
                add_i_e.input_bottom[0],
                dict(text=r"-", text_position="end", text_align="right", move_text=(-0.2, -0.2)),
            ],
        )

        # Outputs of the stage
        outputs = dict(S=pwm.output_right, omega=con_omega.points[1], S_e=pwm_e.output_right[0])

        # Connections to other lines
        connect_to_line = dict(
            epsilon=[
                alpha_beta_to_dq.input_bottom[0],
                dict(text=r"$\varepsilon_{\mathrm{el}}$", text_position="middle", text_align="right"),
            ],
            i=[
                abc_to_alpha_beta.input_right,
                dict(radius=0.1, fill=False, text=[r"$\mathbf{i}_{\mathrm{s a,b,c}}$", "", ""]),
            ],
            i_e=[emf.input_bottom[1], dict(radius=0.05, fill="black")],
        )
        connections = dict()  # Connections

        return start, inputs, outputs, connect_to_line, connections

    return cc_eesm
