from control_block_diagram.components import Box, Circle, Connection, Point, Text
from control_block_diagram.predefined_components import (
    AbcToAlphaBetaTransformation,
    Add,
    AlphaBetaToDqTransformation,
    DqToAbcTransformation,
    Limit,
    PIController,
)


def scim_cc(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the SCIM current control block
    """

    def cc_scim(start, control_task):
        """
        Function to build the SCIM current control block
        Args:
            start:          Starting Point of the Block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # space to the previous block
        space = 1 if control_task == "CC" else 1.5

        # Add blocks for the i_sd and i_sq references and states
        add_i_sd = Add(start.add_x(space))
        add_i_sq = Add(add_i_sd.position.sub(0.5, 1))

        if control_task == "CC":
            # Connections at the inputs
            Connection.connect(
                add_i_sd.input_left[0].sub_x(space),
                add_i_sd.input_left[0],
                text=r"$i^{*}_{\mathrm{sd}}$",
                text_position="start",
                text_align="left",
            )
            Connection.connect(
                add_i_sq.input_left[0].sub_x(space - 0.5),
                add_i_sq.input_left[0],
                text=r"$i^{*}_{\mathrm{sq}}$",
                text_position="start",
                text_align="left",
            )

        # PI Controllers for the d and q component
        pi_i_sd = PIController(
            add_i_sd.position.add_x(1.2), size=(1, 0.8), input_number=1, output_number=1, text="Current\nController"
        )
        pi_i_sq = PIController(
            Point.merge(pi_i_sd.position, add_i_sq.position), size=(1, 0.8), input_number=1, output_number=1
        )

        # Connection between the add blocks and the PI Controllers
        Connection.connect(add_i_sd.output_right, pi_i_sd.input_left)
        Connection.connect(add_i_sq.output_right, pi_i_sq.input_left)

        # Add blocks for the EMF Feedforward
        add_u_sd = Add(pi_i_sd.position.add_x(2))
        add_u_sq = Add(pi_i_sq.position.add_x(1.2))

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
            add_u_sd.output_right[0], dq_to_abc.input_left[0], text=r"$u^{*}_{\mathrm{sd}}$", distance_y=0.28
        )
        Connection.connect(
            add_u_sq.output_right[0],
            dq_to_abc.input_left[1],
            text=r"$u^{*}_{\mathrm{sq}}$",
            distance_y=0.28,
            move_text=(0.4, 0),
        )

        # Limit of the input voltages
        limit = Limit(
            dq_to_abc.position.add_x(2.2),
            size=(1.5, 1.5),
            inputs=dict(left=3, left_space=0.3),
            outputs=dict(right=3, right_space=0.3),
        )

        # Connection between the coordinate transformation and the limit block
        Connection.connect(dq_to_abc.output_right, limit.input_left)

        # Pulse width modulation block
        pwm = Box(
            limit.position.add_x(2.8),
            size=(1.5, 1.2),
            text="PWM",
            inputs=dict(left=3, left_space=0.3),
            outputs=dict(right=3, right_space=0.3),
        )

        # Connection between the limit and the PWM block
        Connection.connect(
            limit.output_right, pwm.input_left, text=[r"$u^*_{\mathrm{s a,b,c}}$", "", ""], distance_y=0.25
        )

        # Coordinate transformation from ABC to AlphaBeta coordinates
        abc_to_alpha_beta = AbcToAlphaBetaTransformation(pwm.position.sub(1, 3), input="right", output="left")

        # Flux observer
        observer = Box(
            abc_to_alpha_beta.position.sub(0.5, 2.2),
            size=(2, 1),
            text="Flux Observer",
            inputs=dict(right=1, top=2, top_space=1.2),
            outputs=dict(left=2),
        )

        # Connection of the flux observer
        con_omega = Connection.connect(observer.input_right[0].add_x(1), observer.input_right[0])
        Connection.connect(
            observer.input_top[0].add_y(0.4),
            observer.input_top[0],
            text=r"$\mathbf{i}_{\mathrm{s a,b,c}}$",
            text_position="start",
        )
        Connection.connect(
            observer.input_top[1].add_y(0.4),
            observer.input_top[1],
            text=r"$\mathbf{u}_{\mathrm{s a,b,c}}$",
            text_position="start",
            move_text=(0, -0.05),
        )

        # Coordinate transformation from AlphaBeta to DQ coordinates
        alpha_beta_to_dq = AlphaBetaToDqTransformation(
            Point.merge(dq_to_abc.position, abc_to_alpha_beta.position), input="right", output="left"
        )

        # Connections between the coordinate transformation and the add blocks
        con_i_sd = Connection.connect(
            alpha_beta_to_dq.output_left[0],
            add_i_sd.input_bottom[0],
            text="-",
            text_position="end",
            text_align="right",
            move_text=(-0.2, -0.2),
        )
        con_i_sq = Connection.connect(
            alpha_beta_to_dq.output_left[1],
            add_i_sq.input_bottom[0],
            text="-",
            text_position="end",
            text_align="right",
            move_text=(-0.2, -0.2),
        )

        # Texts at the previous connections
        Text(position=con_i_sd.start.add(-0.25, 0.25), text=r"$i_{\mathrm{sd}}$")
        Text(position=con_i_sq.start.add(-0.25, 0.25), text=r"$i_{\mathrm{sq}}$")

        # Connection between the flux observer and the coordinate transformation
        Connection.connect(observer.output_left[0], alpha_beta_to_dq.input_bottom[0])

        # Texts at the outputs of the flux observer
        Text(position=observer.output_left[0].add(-0.4, 0.3), text=r"$\angle \hat{\underline{\Psi}}_{\mathrm{r}}$")
        Text(position=observer.output_left[1].add(-0.4, -0.3), text=r"$\hat{\Psi}_{\mathrm{r}}$")

        # Connections between the coordinate transformations
        Connection.connect(
            abc_to_alpha_beta.output,
            alpha_beta_to_dq.input_right,
            text=[r"$i_{\mathrm{s} \upalpha}$", r"$i_{\mathrm{s} \upbeta}$"],
        )
        Connection.connect(
            alpha_beta_to_dq.output_top,
            dq_to_abc.input_bottom,
            text=r"$\angle \hat{\underline{\Psi}}_r$",
            text_align="right",
        )

        # Feedforward block
        feedforward = Box(
            Point.get_mid(add_u_sd.position, add_u_sq.position).sub_y(1.6),
            size=(2, 0.8),
            text="feedforward",
            inputs=dict(bottom=4, bottom_space=0.3),
            outputs=dict(top=2, top_space=0.8),
        )

        # Connections of the feedforward block
        con_omega_2 = Connection.connect(
            feedforward.input_bottom[0].add(7, -4.0702),
            feedforward.input_bottom[0],
            text=r"$\omega_{\mathrm{me}}$",
            text_position="start",
            move_text=(-1, -0.1),
        )
        Connection.connect_to_line(con_omega_2, con_omega.start, arrow=False)
        Connection.connect(feedforward.output_top[0], add_u_sq.input_bottom[0])
        Connection.connect(feedforward.output_top[1], add_u_sd.input_bottom[0])
        con_psi_r = Connection.connect(observer.output_left[1], feedforward.input_bottom[1])
        Connection.connect_to_line(con_i_sd, feedforward.input_bottom[2])
        Connection.connect_to_line(con_i_sq, feedforward.input_bottom[3])

        if control_task in ["TC", "SC"]:
            # Circle at the connection start
            Circle(con_psi_r.points[1], radius=0.05, draw="black", fill="black")
        if control_task == "SC":
            # Circle at the connection start
            Circle(con_omega_2.points[1], radius=0.05, draw="black", fill="black")

        start = pwm.position  # starting point of the next block
        # Inputs of the stage
        inputs = dict(
            i_d_ref=[
                add_i_sd.input_left[0],
                dict(text=r"$i^{*}_{\mathrm{sd}}$", distance_y=0.3, text_position="end", move_text=(-0.85, 0)),
            ],
            i_q_ref=[
                add_i_sq.input_left[0],
                dict(text=r"$i^{*}_{\mathrm{sq}}$", distance_y=0.3, text_position="end", move_text=(-0.35, 0)),
            ],
            omega=[con_omega_2.start, dict(arrow=False)],
        )
        # Outputs of the stage
        outputs = dict(S=pwm.output_right, psi_r=con_psi_r.points[1], omega_me=con_omega_2.points[1])
        # Connections to other lines
        connect_to_line = dict(
            i=[
                abc_to_alpha_beta.input_right,
                dict(radius=0.1, fill=False, text=[r"$\mathbf{i}_{\mathrm{s a,b,c}}$", "", ""]),
            ]
        )
        connections = dict()  # Connections

        return start, inputs, outputs, connect_to_line, connections

    return cc_scim
