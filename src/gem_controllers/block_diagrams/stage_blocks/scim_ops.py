from control_block_diagram.components import Box, Circle, Connection, Path, Point, Text
from control_block_diagram.predefined_components import Add, Divide, IController, Limit, PIController


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
                self.bottom.add_y(self._size_y * by),
                self.top_right.sub(self._size_x * bx, self._size_y * (0.1 + by)),
            ],
            angles=[{"in": 110, "out": -20}, {"in": 200, "out": 70}],
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


def scim_ops(start, control_task):
    """
    Function to build the SCIM operation point selection block
    Args:
        start:          Starting point of the block
        control_task:   Control task of the controller

    Returns:
        endpoint, inputs, outputs, connection to other lines, connections
    """

    start_add = 0  # distance to the start

    # Torque Controller
    if control_task == "TC":
        start_add = 1  # distance to the start

        # Connection at the input
        Connection.connect(
            start, start.add_x(start_add), text="$T^{*}$", text_position="start", text_align="left", arrow=False
        )

    # Limit the torque reference
    box_limit = Limit(start.add_x(start_add + 5), size=(1, 1), inputs=dict(left=1, top=1))

    # Connection between the start and the limit block
    Connection.connect(start.add_x(start_add), box_limit.input_left[0])

    # Optimal flux block with text
    box_psi_opt = PsiOptBox(start.add(start_add + 1, 1.7), size=(1.2, 1))
    Text(position=box_psi_opt.bottom.sub_y(0.3), text=r"$\Psi^{*}_{\mathrm{opt}}(T^{*})$")

    # Minimum flux block
    box_min = Box(box_psi_opt.position.add(1.5, 1.3), inputs=dict(left=2, left_space=0.5), size=(0.8, 1), text="min")

    # Maximum torque block with text
    box_t_max = TMaxPsiBox(box_psi_opt.position.add_x(3.1), size=(1.2, 1))
    Text(position=box_t_max.bottom.sub_y(0.3), text=r"$T_{\mathrm{max}}(\Psi)$")

    # Connection between the start and the optimal flux block
    Connection.connect(start.add_x(start_add), box_psi_opt.input_left[0], start_direction="north")
    Circle(start.add_x(start_add), radius=0.05, fill="black")

    # Connection between the optimal flux and minimum flux block
    Connection.connect(box_psi_opt.output_right[0], box_min.input_left[1])

    # Connection of the maximum torque block
    Connection.connect(box_min.output_right[0].add_x(0.3), box_t_max.input_left[0], start_direction="south")
    Circle(box_min.output_right[0].add_x(0.3), radius=0.05, fill="black")
    Connection.connect(box_t_max.output_right[0], box_limit.input_top[0])

    # Calculation of the i_sq reference
    box_i_sq_ref = Box(
        box_limit.output_right[0].add_x(1.5), size=(1, 0.8), text=r"$\frac{2 L_{\mathrm{r}}}{3 p L_{\mathrm{m}}}$"
    )
    Connection.connect(box_limit.output_right, box_i_sq_ref.input_left, text=r"$T^{*}_{\mathrm{lim}}$")
    divide = Box(
        box_i_sq_ref.output_right[0].add_x(1),
        size=(0.5, 0.5),
        text=r"$\div$",
        inputs=dict(left=1, bottom=1),
        outputs=dict(right=1, top=1),
    )
    Connection.connect(box_i_sq_ref.output_right, divide.input_left)

    # Add block for the flux
    add_psi = Add(box_min.output_right[0].add_x(2.5))

    # Flux Controller
    pi_psi = PIController(add_psi.position.add_x(1.5), text="Flux\nController")

    # Connections of the add block
    Connection.connect(box_min.output_right, add_psi.input_left, text=r"$\Psi^{*}_{\mathrm{lim}}$")
    Connection.connect(divide.output_top, add_psi.input_bottom, text=r"$\hat{\Psi}_{\mathrm{r}}$")
    Connection.connect(add_psi.output_right, pi_psi.input_left)

    # Limit the current reference values
    limit = Limit(
        pi_psi.position.add(3.5, -0.5),
        size=(1.5, 1.5),
        inputs=dict(left=2, left_space=1),
        outputs=dict(right=2, right_space=1),
    )

    # Connections of the limit block
    Connection.connect(pi_psi.output_right[0], limit.input_left[0])
    Connection.connect(divide.output_right[0], limit.input_left[1])

    # Modulation Controller

    # Add the actual flux and delta flux
    add_a = Add(box_min.input_left[0].sub_x(1.5), inputs=dict(left=1, top=1))

    # Connection between the add and minimum block
    Connection.connect(
        add_a.output_right[0],
        box_min.input_left[0],
        text=r"$\Psi_{\mathrm{lim}}$",
        text_align="bottom",
        distance_y=0.25,
    )

    # Calculate the maximum flux
    divide_a = Divide(add_a.position.add_y(1), size=(1, 0.5), inputs="top", input_space=0.5)
    Connection.connect(
        divide_a.output_bottom, add_a.input_top, text=r"$\Psi_{\mathrm{max}}$", text_align="right", distance_x=0.5
    )
    Connection.connect(
        divide_a.input_top[0].add_y(0.3),
        divide_a.input_top[0],
        text=r"$\frac{u_{\mathrm{\mbox{\fontsize{3}{4}\selectfont DC}}}}{\sqrt{3}}$",
        text_position="start",
        text_align="top",
    )
    Connection.connect(
        divide_a.input_top[1].add_y(0.3),
        divide_a.input_top[1],
        text=r"$\omega_{\mathrm{el}}$",
        text_position="start",
        text_align="top",
        move_text=(0, -0.05),
    )

    # Limit the delta flux
    limit_psi = Limit(add_a.input_left[0].sub_x(1.5), size=(1, 1))

    # I Controller of the modulation controller
    i_controller = IController(limit_psi.input_left[0].sub_x(1.2), size=(1.2, 1), text="Modulation\nController")

    # Connections of the limit block
    Connection.connect(i_controller.output_right, limit_psi.input_left)
    Connection.connect(
        limit_psi.output_right[0], add_a.input_left[0], text=r"$\Delta \Psi$", distance_y=0.25, text_align="bottom"
    )

    # Add block for the modulation
    add_a_max = Add(i_controller.position.sub_x(1.5))

    # Connection between the add block and the I Controller
    Connection.connect(add_a_max.output_right, i_controller.input_left)

    # Maximum modulation block
    box_a_max = Box(add_a_max.input_left[0].sub_x(1.3), size=(1.5, 0.8), text=r"$a_{\mathrm{max}} \cdot k$")

    # Connection between the maximum modulation and add block
    Connection.connect(box_a_max.output_right[0], add_a_max.input_left[0], text="$a^{*}$", text_align="bottom")

    # Calculation of the actual modulation
    box_abs = Box(
        add_a_max.input_bottom[0].sub_y(1),
        size=(0.8, 0.8),
        text=r"|\textbf{x}|",
        inputs=dict(bottom=1),
        outputs=dict(top=1),
    )
    con_a = Connection.connect(
        box_abs.output_top[0],
        add_a_max.input_bottom[0],
        text="$-$",
        text_align="right",
        text_position="end",
        move_text=(-0.1, -0.1),
    )
    Text(position=Point.get_mid(*con_a.points).sub_x(0.25), text="$a$")
    div_a = Divide(box_abs.input_bottom[0].sub_y(0.8), size=(1, 0.5), inputs="bottom", input_space=0.5)
    Connection.connect(
        div_a.input_bottom[1].sub_y(0.4),
        div_a.input_bottom[1],
        text=r"$\frac{u_{\mathrm{\mbox{\fontsize{3}{4}\selectfont DC}}}}{2}$",
        text_align="bottom",
        text_position="start",
    )
    Connection.connect(
        div_a.input_bottom[0].sub_y(0.4),
        div_a.input_bottom[0],
        text=r"$\mathbf{u^{*}_{\mathrm{dq}}}$",
        text_align="bottom",
        text_position="start",
    )
    Connection.connect(div_a.output_top, box_abs.input_bottom)

    # Inputs of the stage
    inputs = dict(
        t_ref=[start, dict(arrow=False, text=r"$T^{*}$", end_direction="south", move_text=(-0.25, 1.8))],
        psi_r=[divide.input_bottom[0], dict()],
    )
    outputs = dict(i_q_ref=limit.output_right[1], i_d_ref=limit.output_right[0])  # Outputs of the stage
    connect_to_lines = dict()  # Connections to other lines
    connections = dict()  # Connections of the stage
    start = limit.output_right[0]  # Starting point of the next stage

    return start, inputs, outputs, connect_to_lines, connections
