from gym_electric_motor.core import ElectricMotorVisualization
import gym_electric_motor.visualization.motor_dashboard_plots as mdp
import matplotlib.pyplot as plt


class MotorDashboard(ElectricMotorVisualization):

    def __init__(self, plots, update_cycle=1000):
        plt.ion()
        self._update_cycle = update_cycle
        self._figure = None
        self._plots = []
        for plot in plots:
            if type(plot) is str:
                if plot == 'reward':
                    self._plots.append(mdp.RewardPlot())
                elif plot.startswith('act_'):
                    self._plots.append(mdp.ActionPlot(plot))
                else:
                    self._plots.append(mdp.StatePlot(plot))
            else:
                assert issubclass(plot, mdp.MotorDashboardPlot)
                self._plots.append(plot)

    def reset(self, **__):
        for plot in self._plots:
            plot.reset()

    def step(self, k, state, reference, action, reward, done):
        if not self._figure:
            self._initialize()
        for plot in self._plots:
            plot.step(k, state, reference, action, reward, done)
        if (k + 1) % self._update_cycle == 0:
            self._update()

    def set_modules(self, *modules):
        for plot in self._plots:
            plot.set_modules(*modules)

    def _initialize(self):
        plt.close()
        self._figure, axes = plt.subplots(len(self._plots))
        self._figure.subplots_adjust(wspace=0.0, hspace=0.4)

        for plot, axis in zip(self._plots, axes):
            plot.initialize(axis)
        plt.pause(0.1)

    def _update(self):
        for plot in self._plots:
            plot.update()
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
