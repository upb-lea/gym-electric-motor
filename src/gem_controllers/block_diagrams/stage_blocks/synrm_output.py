from control_block_diagram.components import Connection
from control_block_diagram.predefined_components import DcConverter, SynRM


def synrm_output(emf_feedforward):
    """
    Args:
        emf_feedforward: Boolean whether emf feedforward stage is included

    Returns:
        Function to build the PMSM output block
    """

    def _synrm_output(start, control_task):
        """
        Function to build the PMSM output block
        Args:
            start:          Starting point of the block
            control_task:   Control task of the controller

        Returns:
            endpoint, inputs, outputs, connection to other lines, connections
        """

        # Converter with DC input voltage
        converter = DcConverter(start.add_x(2.7), input_number=3, input_space=0.3, output_number=3)

        # SynRM block
        synrm = SynRM(converter.position.sub_y(5), size=1.3, input="top")

        # Connection between the converter and the motor block
        con_1 = Connection.connect(converter.output_bottom, synrm.input_top, arrow=False)

        start = synrm.position  # starting point of the next block
        # Inputs of the stage
        inputs = dict(S=[converter.input_left, dict(text=[r"$\mathbf{S}_{\mathrm{a,b,c}}$", "", ""], distance_y=0.25)])
        outputs = dict(epsilon=synrm.output_left[0])  # Outputs of the stage
        connect_to_lines = dict()  # Connections to other lines
        connections = dict(i=con_1)  # Connections

        return start, inputs, outputs, connect_to_lines, connections

    return _synrm_output
