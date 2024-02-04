import gem_controllers.stages.base_controllers as bc


def get(base_controller_id: str):
    """Returns the class of a base controller called by a string."""
    return _base_controller_registry[base_controller_id]


_base_controller_registry = {
    "P": bc.PController,
    "I": bc.IController,
    "PI": bc.PIController,
    "PID": bc.PIDController,
    "ThreePoint": bc.ThreePointController,
}
