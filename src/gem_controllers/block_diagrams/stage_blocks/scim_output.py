from control_block_diagram.components import Connection
from control_block_diagram.predefined_components import SCIM, DcConverter


def scim_output(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the SCIM output block
    """

    def _scim_output(start, control_task):
        """
        Function to build the SCIM output block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # Converter with DC input voltage
        converter = DcConverter(start.add_x(2.7), input_number=3, input_space=0.3, output_number=3)

        # SCIM block
        scim = SCIM(converter.position.sub_y(5), size=1.3, input="top")

        # Connection between the converter and the motor block
        con_1 = Connection.connect(converter.output_bottom, scim.input_top, arrow=False)

        start = scim.position  # starting point of the next block
        # Inputs of the stage
        inputs = dict(S=[converter.input_left, dict(text=[r"$\mathbf{S}_{\mathrm{a,b,c}}$", "", ""], distance_y=0.25)])
        outputs = dict(omega=scim.output_left[0])  # Outputs of the stage
        connect_to_lines = dict()  # Connections to other lines
        connections = dict(i=con_1)  # Connections

        return start, inputs, outputs, connect_to_lines, connections

    return _scim_output
