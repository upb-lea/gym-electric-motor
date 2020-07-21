from gym_electric_motor.core import ElectricMotorVisualization
from . import motor_dashboard_plots as mdp
import matplotlib.pyplot as plt


class MotorDashboard(ElectricMotorVisualization):
    """Dashboard to plot the GEM states into graphs.

    Every MotorDashboard consists of multiple MotorDashboardPlots that are each responsible for the plots in a single
    matplotlib axis.
    The MotorDashboard is responsible for all matplotlib.figure related tasks, especially updating the figures.

    """

    def __init__(self, plots, update_cycle=1000, dark_mode=False, **__):
        """
        Args:
            plots(list): A list of plots to show. Each element may either be a string or an already instantiated
                MotorDashboardPlot
                Possible strings:
                    - {state_name}: The corresponding state is plotted
                    - reward: The reward per step is plotted
                    - action_{i}: The i-th action is plotted. 'action_0' for discrete action space
                    - mean_reward: The mean episode reward
                    
            update_cycle(int): Number after how many steps the plot shall be updated. (default 1000)
            dark_mode(Bool):  Select a dark background for visualization by setting it to True
        """
        self._update_cycle = update_cycle
        self._figure = None
        self._figure_ep = None
        self._plots = []
        self._episode_plots = []

        self._dark_mode = dark_mode
        for plot in plots:
            if type(plot) is str:
                if plot == 'reward':
                    self._plots.append(mdp.RewardPlot())
                elif plot.startswith('action_'):
                    self._plots.append(mdp.ActionPlot(plot))
                elif plot.startswith('mean_reward'):
                    self._episode_plots.append(mdp.MeanEpisodeRewardPlot())
                else:
                    self._plots.append(mdp.StatePlot(plot))
            else:
                assert issubclass(plot, mdp.MotorDashboardPlot)
                self._plots.append(plot)

    def reset(self, **__):
        """Called when the environment is reset. All subplots are reset.
        """

        for plot in self._plots:  # for plot in self._plots + self._episode_plots throws warning
            plot.reset()
        for plot in self._episode_plots:
            plot.reset()

        # since end of an episode can only be identified by a reset call. Episode based plot canvas updated here
        if self._figure_ep:
            self._figure_ep.canvas.draw()
            self._figure_ep.canvas.flush_events()

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
        if not (self._figure or self._figure_ep):
            self._initialize()
        for plot in self._plots :    # for plot in self._plots + self._episode_plots throws warning
            plot.step(k, state, reference, action, reward, done)
        for plot in self._episode_plots:
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
        for plot in self._plots:  # for plot in self._plots + self._episode_plots throws warning
            plot.set_modules(ps, rg, rf)
        for plot in self._episode_plots:
            plot.set_modules(ps, rg, rf)

    def _initialize(self):
        """Called with first render() call to setup the figures and plots.
        """
        axis_ep = []
        plt.close()
        assert len(self._plots) > 0, "no plot variable selected"
        # Use dark-mode, if selected
        if self._dark_mode:
            plt.style.use('dark_background')
        # create seperate figures for time based and episode based plots
        self._figure, axes = plt.subplots(len(self._plots), sharex=True)
        if self._episode_plots:
            self._figure_ep, axes_ep = plt.subplots(len(self._episode_plots))
            self._figure_ep.subplots_adjust(wspace=0.0, hspace=0.02)
            self._figure.subplots_adjust(wspace=0.0, hspace=0.2)
            self._figure_ep.text(0.5, 0.04, 'episode', va='center', ha='center')

        # adding a common x-label to all the subplots in each figure
        self._figure.text(0.5, 0.04, 't/s', va='center', ha='center')

        # plt.subplot() does not return an iterable var when the number of subplots==1
        if len(self._plots) == 1:
            axes = [axes]
        if  len(self._episode_plots) == 1:
            axis_ep = [axes_ep]
        for plot, axis in zip(self._plots, axes):
            plot.initialize(axis)

        for plot, axis in zip(self._episode_plots, axis_ep):
            plot.initialize(axis)
        plt.pause(0.1)

    def _update(self):
        """Called every {update_cycle} steps to refresh the figure.
        """
        for plot in self._plots:
            plot.update()

        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
