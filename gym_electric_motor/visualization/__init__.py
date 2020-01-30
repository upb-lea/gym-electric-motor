from .console_printer import ConsolePrinter
from .motor_dashboard import MotorDashboard

from ..utils import register_class
from .. import ElectricMotorVisualization

register_class(ConsolePrinter, ElectricMotorVisualization, 'ConsolePrinter')
register_class(MotorDashboard, ElectricMotorVisualization, 'MotorDashboard')