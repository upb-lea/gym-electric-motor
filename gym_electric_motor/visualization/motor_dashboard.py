from gym_electric_motor.core import ElectricMotorVisualization
import gym_electric_motor.visualization.motor_dashboard_plots as mdp
import matplotlib
import matplotlib.pyplot as plt


class MotorDashboard(ElectricMotorVisualization):
    """Dashboard to plot the GEM states into graphs.

    Every MotorDashboard consists of multiple MotorDashboardPlots that are each responsible fpr the plots in a single
    matplotlib axis.
    The MotorDashboard is responsible for all matplotlib.figure related tasks, especially updating the figures.

    """

    def __init__(self, plots, update_cycle=1000):
        """
        Args:
            plots(list): A list of plots to show. Each element may either be a string or an already instantiated
                MotorDashboardPlot
                Possible strings:
                    - {state_name}: The corresponding state is plotted
                    - reward: The reward per step is plotted
                    - action_{i}: The i-th action is plotted (might be refined)
            update_cycle(int): Number after how many steps the plot shall be updated. (default 1000)
        """
        plt.ion()
        self._update_cycle = update_cycle
        self._figure = None
        self._plots = []
        for plot in plots:
            if type(plot) is str:
                if plot == 'reward':
                    self._plots.append(mdp.RewardPlot())
                elif plot.startswith('action_'):
                    self._plots.append(mdp.ActionPlot(plot))
                else:
                    self._plots.append(mdp.StatePlot(plot))
            else:
                assert issubclass(plot, mdp.MotorDashboardPlot)
                self._plots.append(plot)

    def reset(self, **__):
        """Called when the environment is reset. All subplots are reset.
        """
        for plot in self._plots:
            plot.reset()

    def step(self, k, state, reference, action, reward, done):
        """ Called within a render() call of an environment.

        The information about the last environmental step is passed.

        Args:
            k(int): Current episode step.
            state(ndarray(float)): State of the system
            reference(ndarray(float)): Reference array of the system
            action(ndarray(float)): Last taken action. (None after reset)
            reward(ndarray(float)): Last received reward. (None after reset)
            done(bool): Flag if the current state is terminal
        """
        if not self._figure:
            self._initialize()
        for plot in self._plots:
            plot.step(k, state, reference, action, reward, done)
        if (k + 1) % self._update_cycle == 0:
            self._update()

    def set_modules(self, ps, rg, rf):
        """Called during initialization of the environment to interconnect all modules. State_names, references,...
        might be saved here for later processing

        Args:
            ps(PhysicalSystem): PhysicalSystem of the environment
            rg(ReferenceGenerator): ReferenceGenerator of the environment
            rf(RewardFunction): RewardFunction of the environment
        """
        for plot in self._plots:
            plot.set_modules(*modules)

    def _initialize(self):
        """Called with first render() call to setup the figures and plots.
        """
        plt.close()
        self._figure, axes = plt.subplots(len(self._plots))
        self._figure.subplots_adjust(wspace=0.0, hspace=0.4)

        for plot, axis in zip(self._plots, axes):
            plot.initialize(axis)
        plt.pause(0.1)

    def _update(self):
        """Called every {update_cycle} steps to refresh the figure.
        """
        for plot in self._plots:
            plot.update()
        if matplotlib.get_backend() == 'NbAgg':
            self._figure.canvas.draw()
        self._figure.canvas.flush_events()
