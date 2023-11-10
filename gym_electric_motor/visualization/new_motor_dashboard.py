from gym_electric_motor.core import ElectricMotorVisualization
from .motor_dashboard_plots import StatePlot, ActionPlot, RewardPlot, TimePlot, EpisodePlot, StepPlot
from .motor_dashboard import MotorDashboard
import matplotlib.pyplot as plt
import gymnasium


class NewMotorDashboard(MotorDashboard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.on_reset_begin = None
