from control_block_diagram import ControllerDiagram
from control_block_diagram.components import Connection, Point

import gem_controllers as gc

from .stage_blocks import (
    eesm_cc,
    eesm_ops,
    eesm_output,
    ext_ex_dc_cc,
    ext_ex_dc_ops,
    ext_ex_dc_output,
    perm_ex_dc_cc,
    perm_ex_dc_ops,
    perm_ex_dc_output,
    pi_speed_controller,
    pmsm_cc,
    pmsm_ops,
    pmsm_output,
    pmsm_speed_controller,
    scim_cc,
    scim_ops,
    scim_output,
    scim_speed_controller,
    series_dc_cc,
    series_dc_ops,
    series_dc_output,
    shunt_dc_cc,
    shunt_dc_ops,
    shunt_dc_output,
    synrm_cc,
    synrm_output,
)


def build_block_diagram(controller, env_id, save_block_diagram_as):
    """
    Creates a block diagram of the controller

    Args:
        controller:             GEMController
        env_id:                 GEM environment id
        save_block_diagram_as:  string or tuple of strings of the data types to be saved

    Returns:
        Control Block Diagram

    """

    # Get the block building function for all stages
    motor_type = gc.utils.get_motor_type(env_id)
    control_task = gc.utils.get_control_task(env_id)
    stages = get_stages(controller.controller, motor_type)

    # Create a new block diagram
    doc = ControllerDiagram()

    # Help parameter

    start = Point(0, 0)
    inputs = dict()
    outputs = dict()
    connections = dict()
    connect_to_lines = dict()

    # Build the stage blocks
    for stage in stages:
        start, inputs_, outputs_, connect_to_lines_, connections_ = stage(start, control_task)
        inputs = {**inputs, **inputs_}
        outputs = {**outputs, **outputs_}
        connect_to_lines = {**connect_to_lines, **connect_to_lines_}
        connections = {**connections, **connections_}

    # Connect the different blocks
    for key in inputs.keys():
        if key in outputs.keys():
            connections[key] = Connection.connect(outputs[key], inputs[key][0], **inputs[key][1])

    for key in connect_to_lines.keys():
        if key in connections.keys():
            Connection.connect_to_line(connections[key], connect_to_lines[key][0], **connect_to_lines[key][1])

    # Save the block diagram
    if save_block_diagram_as is not None:
        save_block_diagram_as = (
            list(save_block_diagram_as) if isinstance(save_block_diagram_as, (tuple, list)) else [save_block_diagram_as]
        )
        doc.save(*save_block_diagram_as)

    return doc


def get_stages(controller, motor_type):
    """
    Function to get all block building functions

    Args:
        controller: GEMController
        motor_type: type of the motor

    Returns:
        list of all block building functions

    """

    motor_check = motor_type in ["PMSM", "SCIM", "EESM", "SynRM", "ShuntDc", "SeriesDc", "PermExDc", "ExtExDc"]
    stages = []

    # Add the speed controller block function
    if isinstance(controller, gc.PISpeedController):
        if motor_type in ["PMSM", "SCIM", "EESM", "SynRM"]:
            stages.append(build_functions[motor_type + "_Speed_Controller"])
        else:
            stages.append(build_functions["PI_Speed_Controller"])
        controller = controller.torque_controller

    # add the torque controller block function
    if isinstance(controller, gc.torque_controller.TorqueController):
        if motor_check:
            stages.append(build_functions[motor_type + "_OPS"])
        controller = controller.current_controller

    # add the current controller block function
    if isinstance(controller, gc.PICurrentController):
        emf_feedforward = controller.emf_feedforward is not None
        if motor_check:
            stages.append(build_functions[motor_type + "_CC"](emf_feedforward))

    # add the output block function
    stages.append((build_functions[motor_type + "_Output"](controller.emf_feedforward is not None)))

    return stages


# dictonary of all block building functions
build_functions = {
    "PI_Speed_Controller": pi_speed_controller,
    "PMSM_Speed_Controller": pmsm_speed_controller,
    "SCIM_Speed_Controller": scim_speed_controller,
    "EESM_Speed_Controller": pmsm_speed_controller,
    "SynRM_Speed_Controller": pmsm_speed_controller,
    "PMSM_OPS": pmsm_ops,
    "SCIM_OPS": scim_ops,
    "EESM_OPS": eesm_ops,
    "SynRM_OPS": pmsm_ops,
    "SeriesDc_OPS": series_dc_ops,
    "ShuntDc_OPS": shunt_dc_ops,
    "PermExDc_OPS": perm_ex_dc_ops,
    "ExtExDc_OPS": ext_ex_dc_ops,
    "PMSM_CC": pmsm_cc,
    "SCIM_CC": scim_cc,
    "EESM_CC": eesm_cc,
    "SynRM_CC": synrm_cc,
    "SeriesDc_CC": series_dc_cc,
    "ShuntDc_CC": shunt_dc_cc,
    "PermExDc_CC": perm_ex_dc_cc,
    "ExtExDc_CC": ext_ex_dc_cc,
    "PMSM_Output": pmsm_output,
    "SCIM_Output": scim_output,
    "EESM_Output": eesm_output,
    "SynRM_Output": synrm_output,
    "SeriesDc_Output": series_dc_output,
    "ShuntDc_Output": shunt_dc_output,
    "PermExDc_Output": perm_ex_dc_output,
    "ExtExDc_Output": ext_ex_dc_output,
}
