from control_block_diagram.components import Box, Circle, Connection, Path, Point, Text
from control_block_diagram.predefined_components import Add, Divide, IController, Limit


class PsiOptBox(Box):
    """Optimal flux block"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # border in x- and y-direction
        bx = 0.1
        by = 0.15

        # Coordinate system
        Connection.connect(
            self.bottom_left.add(self._size_x * bx, self._size_y * by),
            self.bottom_right.add(-self._size_x * bx, self._size_y * by),
        )
        Connection.connect(self.bottom.add_y(self._size_y * by), self.top.sub_y(self._size_y * by))

        # Graph in the coordinate system
        Path(
            [
                self.top_left.add(self._size_x * bx, -self._size_y * (0.1 + by)),
                self.bottom.add_y(self._size_y * (0.2 + by)),
                self.top_right.sub(self._size_x * bx, self._size_y * (0.1 + by)),
            ],
            angles=[{"in": 180, "out": 0}, {"in": 180, "out": 0}],
            arrow=False,
        )


class TMaxPsiBox(Box):
    """Maximum torque block"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # border in x- and y-direction
        bx = 0.1 * self._size_x
        by = 0.15 * self._size_y
        scale = 1.5

        # Coordinate system
        Connection.connect(self.bottom_left.add(bx, by), self.top_left.add(bx, -by))
        Connection.connect(self.left.add_x(bx), self.right.sub_x(bx))

        # Graph in the coordinate system
        Path(
            [self.left.add_x(bx), self.top_right.sub(scale * bx, scale * by)],
            arrow=False,
            angles=[{"in": 180, "out": 35}],
        )
        Path(
            [self.left.add_x(bx), self.bottom_right.add(-scale * bx, scale * by)],
            arrow=False,
            angles=[{"in": 180, "out": -35}],
        )


def eesm_ops(start, control_task):
    """
    Function to build the EESM operation point selection block
    Args:
        start:          Starting point of the block
        control_task:   Control task of the controller

    Returns:
        endpoint, inputs, outputs, connection to other lines, connections
    """

    if control_task == "TC":
        # Connection at the input
        Connection.connect(start, start.add_x(1), text="$T^{*}$", text_position="start", text_align="left", arrow=False)

    # Limit the torque reference
    box_limit = Limit(start.add_x(6), inputs=dict(left=1, bottom=1), size=(1, 1))

    # Calculate the optimal flux
    box_psi_opt = PsiOptBox(start.add(2, -1.7), size=(1.2, 1))
    Text(position=box_psi_opt.top.add_y(0.25), text=r"$\Psi^{*}_{\mathrm{opt}}(T^{*})$")

    # Minimum block
    box_min = Box(box_psi_opt.position.add(1.5, -1.3), inputs=dict(left=2, left_space=0.5), size=(0.8, 1), text="min")

    # Calculate the maximum torque
    box_t_max = TMaxPsiBox(box_psi_opt.position.add_x(3.1), size=(1.2, 1))
    Text(position=box_t_max.top.add_y(0.25), text=r"$T_{\mathrm{max}}(\Psi)$")

    # Connect the maximum torque and limit block
    con_torque = Connection.connect(start.add_x(1), box_limit.input_left[0])

    # Connection between the input and the optimum flux block
    Connection.connect(start.add_x(1), box_psi_opt.input_left[0], start_direction="south")
    Circle(start.add_x(1), radius=0.05, fill="black")

    # Connection between the optimum flux and minimum block
    Connection.connect(box_psi_opt.output_right[0], box_min.input_left[0])

    # Conncetion between the minimum and maximum torque block
    Connection.connect(box_min.output_right[0].add_x(0.3), box_t_max.input_left[0], start_direction="north")

    # Circle at the connection
    Circle(box_min.output_right[0].add_x(0.3), radius=0.05, fill="black")

    # Conncetion between the maximum torque and torque limit block
    Connection.connect(box_t_max.output_right[0], box_limit.input_bottom[0])

    # Optimal operation point function block
    box_f_psi_t = Box(
        box_limit.position.add(2.2, -1.5),
        size=(1.3, 3.5),
        inputs=dict(left=2, left_space=3),
        outputs=dict(right=3, right_space=1),
        text=r"\textbf{f}($\Psi$, $T$)",
    )

    # Conncetions to the optimal opertion point function block
    Connection.connect(box_min.output_right[0], box_f_psi_t.input_left[1], text=r"$\Psi^{*}_{\mathrm{lim}}$")
    Connection.connect(box_limit.output_right[0], box_f_psi_t.input_left[0], text=r"$T^{*}_{\mathrm{lim}}$")

    # Moulation Controller

    # Add the actual flux and delta flux
    add_psi = Add(box_min.input_left[1].sub_x(1))

    # Connection between the add and minimum block
    Connection.connect(
        add_psi.output_right[0], box_min.input_left[1], text=r"$\Psi_{\mathrm{lim}}$", text_align="top", distance_y=0.25
    )

    # Limit the delta flux
    limit_modulation = Limit(add_psi.input_left[0].sub_x(1.5), size=(1, 1))

    # I Controller of the modulation controller
    i_controller = IController(limit_modulation.input_left[0].sub_x(1), size=(1.2, 1), text="Modulation\nController")

    # Connections of the limit block
    Connection.connect(i_controller.output_right, limit_modulation.input_left)
    Connection.connect(limit_modulation.output_right[0], add_psi.input_left[0], text=r"$\Delta \Psi$", distance_y=0.25)

    # Add block of the modulation controller
    add_a = Add(i_controller.position.sub_x(1.5))

    # Maximum modulation
    box_a_max = Box(add_a.input_left[0].sub_x(1.3), size=(1.5, 0.8), text=r"$a_{\mathrm{max}} \cdot k$")

    # Building the absolute modulation
    box_abs = Box(
        add_a.position.sub_y(1.2), size=(0.8, 0.8), text=r"|\textbf{x}|", inputs=dict(bottom=1), outputs=dict(top=1)
    )

    # Conncetions of the add block
    Connection.connect(box_a_max.output_right[0], add_a.input_left[0], text="$a^{*}$")
    Connection.connect(add_a.output_right[0], i_controller.input_left[0])
    con_a = Connection.connect(
        box_abs.output_top[0],
        add_a.input_bottom[0],
        text="$-$",
        text_align="right",
        text_position="end",
        move_text=(-0.1, -0.2),
    )

    # Additional text at the connection
    Text(position=Point.get_mid(*con_a.points).sub_x(0.25), text="$a$")

    # divide the actual voltage by the dc voltage
    div_a = Divide(box_abs.input_bottom[0].sub_y(1), size=(1, 0.5), inputs="bottom", input_space=0.5)

    # Conncetions of the divide block
    Connection.connect(
        div_a.input_bottom[0].sub_y(0.7),
        div_a.input_bottom[0],
        text=r"$\mathbf{u^{*}_{\mathrm{dq}}}$",
        text_position="start",
        text_align="bottom",
    )
    Connection.connect(
        div_a.input_bottom[1].sub_y(0.7),
        div_a.input_bottom[1],
        text=r"$\frac{u_{\mathrm{\mbox{\fontsize{3}{4}\selectfont DC}}}}{2}$",
        text_position="start",
        text_align="bottom",
    )
    Connection.connect(div_a.output_top[0], box_abs.input_bottom[0])

    # Divide the dc voltage by the electrical speed
    div_psi = Divide(add_psi.input_bottom[0].sub_y(2), size=(1, 0.5), inputs="bottom", input_space=0.5)

    # Connections of the divide block
    Connection.connect(
        div_psi.output_top[0],
        add_psi.input_bottom[0],
        text=r"$\Psi_{\mathrm{max}}$",
        text_align="right",
        distance_x=0.5,
    )
    Connection.connect(
        div_psi.input_bottom[0].sub_y(0.7),
        div_psi.input_bottom[0],
        text=r"$\frac{u_{\mathrm{\mbox{\fontsize{3}{4}\selectfont DC}}}}{\sqrt{3}}$",
        text_position="start",
        text_align="bottom",
    )

    # Limit of the current reference values
    limit = Limit(
        Point.get_mid(box_f_psi_t.output_right[1], box_f_psi_t.output_right[2]).add_x(1),
        size=(1, 1.5),
        inputs=dict(left=2, left_space=1),
        outputs=dict(right=2, right_space=1),
    )
    limit_e = Limit(box_f_psi_t.output_right[0].add_x(1), size=(1, 1), inputs=dict(left=1), outputs=dict(right=1))

    # Connections between the optimal opertion point function and limit block
    Connection.connect(box_f_psi_t.output_right[0], limit_e.input_left[0])
    Connection.connect(box_f_psi_t.output_right[1], limit.input_left[0])
    Connection.connect(box_f_psi_t.output_right[2], limit.input_left[1])

    # Inputs of the stage
    inputs = dict(
        t_ref=[con_torque.start, dict(arrow=False, text=r"$T^{*}$")],
        omega=[div_psi.input_bottom[1], dict(text=r"$\omega_{\mathrm{el}}$", move_text=(3, 0))],
    )
    # Outputs of the stage
    outputs = dict(i_d_ref=limit.output_right[0], i_q_ref=limit.output_right[1], i_e_ref=limit_e.output_right[0])
    connect_to_lines = dict()  # Connections to other lines
    connections = dict()  # Connections of the stage
    start = limit.output_right[0]  # Starting point of the next stage

    return start, inputs, outputs, connect_to_lines, connections
