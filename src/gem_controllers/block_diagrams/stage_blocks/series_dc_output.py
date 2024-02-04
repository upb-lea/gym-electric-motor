from control_block_diagram.components import Box, Connection
from control_block_diagram.predefined_components import DcConverter, DcSeriesMotor, Limit


def series_dc_output(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the Series DC output block
    """

    def _series_dc_output(start, control_task):
        """
        Function to build the Series DC output block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # space to the previous block
        space = 1.5 if emf_feedforward else 2

        # voltage limit block
        limit = Limit(start.add_x(space), size=(1, 1))

        # pulse width modulation block
        pwm = Box(limit.output_right[0].add_x(1.5), size=(1, 0.8), text="PWM")

        # connection between limit and pwm block
        Connection.connect(limit.output_right, pwm.input_left)

        # Converter with DC input voltage
        converter = DcConverter(pwm.position.add_x(2), size=1.2, input_number=1)

        # Connection between pwm and converter block
        Connection.connect(pwm.output_right, converter.input_left, text="$S$")

        # DC Series motor block
        dc_series = DcSeriesMotor(converter.position.sub_y(3), size=1.2, input="top", output="left")

        # Connections between converter and motor block
        conv_motor = Connection.connect(
            converter.output_bottom, dc_series.input_top, text=["", r"$i$"], text_align="right", arrow=False
        )

        # Connection between previous connection and current output
        con_i = Connection.connect_to_line(
            conv_motor[0], pwm.position.sub_y(1.5), text=r"$i$", arrow=False, fill=False, radius=0.1
        )

        start = converter.position  # starting point of the next block
        inputs = dict(u=[limit.input_left[0], dict(text=r"$u^{*}$")])  # Inputs of the stage
        outputs = dict(i=con_i.end)  # Outputs of the stage
        connect_to_lines = dict()  # Connections to other lines
        connections = dict()  # Connections

        if emf_feedforward or control_task in ["SC"]:
            # Connection between the motor output and the omega output of the stage
            con_omega = Connection.connect(
                dc_series.output_left[0].sub_x(2), dc_series.output_left[0], text=r"$\omega_{\mathrm{me}}$", arrow=False
            )
            # Add the omega output
            outputs["omega"] = con_omega.end

        return start, inputs, outputs, connect_to_lines, connections

    return _series_dc_output
