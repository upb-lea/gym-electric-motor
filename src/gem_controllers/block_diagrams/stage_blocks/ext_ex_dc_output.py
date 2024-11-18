from control_block_diagram.components import Connection
from control_block_diagram.predefined_components import DcConverter, DcExtExMotor


def ext_ex_dc_output(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the Externally Excited DC output block
    """

    def _ext_ex_dc_output(start, control_task):
        """
        Function to build the Externally Excited DC output block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # Converter with DC input voltage and four outputs
        converter = DcConverter(
            start.add(2.5, -0.6), size=1.7, input_number=2, input_space=1.2, output_number=4, output_space=0.25
        )

        # Ext Ex DC motor block
        dc_ext_ex = DcExtExMotor(converter.position.sub_y(3), size=1.2, input="top", output="left")

        # Connections between converter and motor block
        con_conv_a = Connection.connect(converter.output_bottom[0], dc_ext_ex.input_top[0], arrow=False)
        Connection.connect(converter.output_bottom[1], dc_ext_ex.input_top[1], arrow=False)
        con_conv_e = Connection.connect(converter.output_bottom[2], dc_ext_ex.input_top[2], arrow=False)
        Connection.connect(converter.output_bottom[3], dc_ext_ex.input_top[3], arrow=False)

        # Connection to the i_e output
        con_e = Connection.connect_to_line(
            con_conv_e, start.sub_y(2), arrow=False, radius=0.1, fill=False, text=r"$i_{\mathrm{e}}$", distance_y=0.25
        )

        # Connection to the i_a output
        con_a = Connection.connect_to_line(
            con_conv_a,
            start.sub_y(2.5),
            arrow=False,
            radius=0.1,
            fill=False,
            text=r"$i_{\mathrm{a}}$",
            distance_y=0.25,
            text_align="bottom",
            move_text=(0.25, 0),
        )

        start = converter.position  # starting point of the next block
        # Inputs of the stage
        inputs = dict(
            u_e=[converter.input_left[0], dict(text=r"$\mathrm{S_e}$", distance_y=0.25)],
            u_a=[converter.input_left[1], dict(text=r"$\mathrm{S_a}$", distance_y=0.25)],
        )
        outputs = dict(i_e=con_e.end, i_a=con_a.end)  # Outputs of the stage
        connect_to_lines = dict()  # Connections to other lines
        connections = dict()  # Connections

        if emf_feedforward or control_task in ["SC"]:
            # Connection between the motor output and the omega output of the stage
            con_omega = Connection.connect(
                dc_ext_ex.output_left[0].sub_x(2), dc_ext_ex.output_left[0], text=r"$\omega_{\mathrm{me}}$", arrow=False
            )
            # Add the omega output
            outputs["omega"] = con_omega.end

        return start, inputs, outputs, connect_to_lines, connections

    return _ext_ex_dc_output
