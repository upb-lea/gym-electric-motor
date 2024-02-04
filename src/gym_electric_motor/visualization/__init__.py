from ..core import ElectricMotorVisualization
from ..utils import register_class
from .console_printer import ConsolePrinter
from .motor_dashboard import MotorDashboard
from .render_modes import RenderMode

register_class(ConsolePrinter, ElectricMotorVisualization, "ConsolePrinter")
register_class(MotorDashboard, ElectricMotorVisualization, "MotorDashboard")
