from gym_electric_motor.core import ElectricMotorVisualization
from .motor_dashboard_plots import StatePlot, ActionPlot, RewardPlot, TimePlot, EpisodePlot, StepPlot
import matplotlib.pyplot as plt
import gym


class MotorDashboard(ElectricMotorVisualization):
    """A dashboard to plot the GEM states into graphs.

    Every MotorDashboard consists of multiple MotorDashboardPlots that are each responsible for the plots in a single
    matplotlib axis.

    It handles three different types of plots: The TimePlot, EpisodePlot and StepPlot which especially differ in
    their x-Axis. The time plots plot every step and have got the time on the x-Axis. The EpisodicPlots plot statistics
    over the episodes (e.g. mean reward per step in each episode). The episode number is on their x-Axis. The
    StepPlots plot statistics over the last taken steps (e.g. mean reward over the last 1000 steps) and their x-Axis
    are the cumulative number of steps.

    The StepPlots, EpisodicPlots and TimePlots each are plotted into three separate figures.

    The most common TimePlots (i.e to plot the states, actions and rewards) can be plotted by just passing the
    corresponding arguments in the constructor. Additional plots (e.g. the MeanEpisodeRewardPlot) have to be
    initialized manually and passed to the constructor.

    Furthermore, completely custom plots can be defined. They have to derive from the TimePlot, EpisodePlot or
    StepPlot base classes.
    """

    def __init__(
        self, state_plots=(), action_plots=(), reward_plot=False, additional_plots=(),
        update_interval=1000, time_plot_width=10000, style=None
    ):
        """
        Args:
            state_plots('all'/iterable(str)): An iterable of state names to be shown. If 'all' all states will be shown.
                Default: () (no plotted states)
            action_plots('all'/iterable(int)): If action_plots='all', all actions will be plotted. If more than one
                action can be applied on the environment it can be selected by its index.
                Default: () (no plotted actions).
            reward_plot(boolean): Select if the current reward is to be plotted. Default: False
            additional_plots(iterable((TimePlot/EpisodePlot/StepPlot))): Additional already instantiated plots
                to be shown on the dashboard
            update_interval(int > 0): Amount of steps after which the plots are updated. Updating each step reduces the
                performance drastically. Default: 1000
            time_plot_width(int > 0): Width of the step plots in steps. Default: 10000 steps
                (1 second for continuously controlled environments / 0.1 second for discretely controlled environments)
            style(string): Select one of the matplotlib-styles. e.g. "dark-background".
                Default: None (the already selected style)
        """
        # Basic assertions
        assert type(reward_plot) is bool
        assert all(isinstance(ap, (TimePlot, EpisodePlot, StepPlot)) for ap in additional_plots)
        assert type(update_interval) in [int, float]
        assert update_interval > 0
        assert type(time_plot_width) in [int, float]
        assert time_plot_width > 0
        assert style in plt.style.available or style is None

        super().__init__()

        # Select the matplotlib style
        if style is not None:
            plt.style.use(style)
        # List of the opened figures
        self._figures = []
        # The figures to be opened for the step plots, episodic plots and step plots
        self._time_plot_figure = None
        self._episodic_plot_figure = None
        self._step_plot_figure = None

        # Store the input data
        self._state_plots = state_plots
        self._action_plots = action_plots
        self._reward_plot = reward_plot

        # Separate the additional plots into StepPlots, EpisodicPlots and StepPlots
        self._custom_time_plots = [p for p in additional_plots if isinstance(p, TimePlot)]
        self._episodic_plots = [p for p in additional_plots if isinstance(p, EpisodePlot)]
        self._step_plots = [p for p in additional_plots if isinstance(p, StepPlot)]

        self._time_plots = []
        self._update_interval = int(update_interval)
        self._time_plot_width = int(time_plot_width)
        self._plots = []
        self._k = 0
        self._update_render = False

    def on_reset_begin(self):
        """Called before the environment is reset. All subplots are reset."""
        for plot in self._plots:
            plot.on_reset_begin()

    def on_reset_end(self, state, reference):
        """Called after the environment is reset. The initial data is passed.

        Args:
            state(array(float)): The initial state :math:`s_0`.
            reference(array(float)): The initial reference for the first time step :math:`s^*_0`.
        """
        for plot in self._plots:
            plot.on_reset_end(state, reference)

    def on_step_begin(self, k, action):
        """The information about the last environmental step is passed.

        Args:
            k(int): The current episode step.
            action(ndarray(float) / int): The taken action :math:`a_k`.
        """
        for plot in self._plots:
            plot.on_step_begin(k, action)

    def on_step_end(self, k, state, reference, reward, done):
        """The information after the step is passed

        Args:
            k(int): The current episode step
            state(array(float)): The state of the env after the step :math:`s_k`.
            reference(array(float)): The reference corresponding to the state :math:`s^*_k`.
            reward(float): The reward that has been received for the last action that lead to the current state
                :math:`r_{k}`.
            done(bool): Flag, that indicates, if the last action lead to a terminal state :math:`t_{k}`.
        """
        for plot in self._plots:
            plot.on_step_end(k, state, reference, reward, done)
        self._k += 1
        if self._k % self._update_interval == 0:
            self._update_render = True

    def render(self):
        """Updates the plots every *update cycle* calls of this method."""
        if not (self._time_plot_figure or self._episodic_plot_figure or self._step_plot_figure) \
           and len(self._plots) > 0:
            self._initialize()
        if self._update_render:
            self._update()
            self._update_render = False

    def set_env(self, env):
        """Called during initialization of the environment to interconnect all modules. State names, references,...
        might be saved here for later processing

        Args:
            env(ElectricMotorEnvironment): The environment.
        """
        state_names = env.physical_system.state_names
        if self._state_plots == 'all':
            self._state_plots = state_names
        if self._action_plots == 'all':
            if type(env.action_space) is gym.spaces.Discrete:
                self._action_plots = [0]
            elif type(env.action_space) in (gym.spaces.Box, gym.spaces.MultiDiscrete):
                self._action_plots = list(range(env.action_space.shape[0]))

        self._time_plots = []

        if len(self._state_plots) > 0:
            assert all(state in state_names for state in self._state_plots)
            for state in self._state_plots:
                self._time_plots.append(StatePlot(state))

        if len(self._action_plots) > 0:
            assert type(env.action_space) in (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete), \
                f'Action space of type {type(env.action_space)} not supported for plotting.'
            for action in self._action_plots:
                ap = ActionPlot(action)
                self._time_plots.append(ap)

        if self._reward_plot:
            self._reward_plot = RewardPlot()
            self._time_plots.append(self._reward_plot)

        self._time_plots += self._custom_time_plots

        self._plots = self._time_plots + self._episodic_plots + self._step_plots

        for time_plot in self._time_plots:
            time_plot.set_width(self._time_plot_width)

        for plot in self._plots:
            plot.set_env(env)

    def reset_figures(self):
        """Method to reset the figures to the initial state.

        This method can be called, when the plots shall be reset after the training and before the test, for example.
        Another use case, that requires the call of this method by the user, is when the dashboard is executed within
        a jupyter notebook and the figures shall be plotted below a new cell."""
        for plot in self._plots:
            plot.reset_data()
        self._episodic_plot_figure = self._time_plot_figure = self._step_plot_figure = None
        self._figures = []

    def _initialize(self):
        """Called with first render() call to setup the figures and plots."""
        plt.close()
        self._figures = []

        if plt.get_backend() == 'nbAgg':
            self._initialize_figures_notebook()
        else:
            self._initialize_figures_window()

        plt.pause(0.1)

    def _initialize_figures_notebook(self):
        # Create all plots below each other: First Time then Episode then Step Plots
        no_of_plots = len(self._episodic_plots) + len(self._step_plots) + len(self._time_plots)

        if no_of_plots == 0:
            return
        fig, axes = plt.subplots(no_of_plots, figsize=(8, 2*no_of_plots))
        self._figures = [fig]
        axes = [axes] if no_of_plots == 1 else axes
        time_axes = axes[:len(self._time_plots)]
        axes = axes[len(self._time_plots):]
        if len(self._time_plots) > 0:
            time_axes[-1].set_xlabel('t/s')
            self._time_plot_figure = fig
            for plot, axis in zip(self._time_plots, time_axes):
                plot.initialize(axis)
        episode_axes = axes[:len(self._episodic_plots)]
        axes = axes[len(self._episodic_plots):]
        if len(self._episodic_plots) > 0:
            episode_axes[-1].set_xlabel('Episode No')
            self._episodic_plot_figure = fig
            for plot, axis in zip(self._episodic_plots, episode_axes):
                plot.initialize(axis)
        step_axes = axes
        if len(self._step_plots) > 0:
            step_axes[-1].set_xlabel('Cumulative Steps')
            self._step_plot_figure = fig
            for plot, axis in zip(self._step_plots, step_axes):
                plot.initialize(axis)

    def _initialize_figures_window(self):
        # create separate figures for time based, step and episode based plots
        if len(self._episodic_plots) > 0:
            self._episodic_plot_figure, axes_ep = plt.subplots(len(self._episodic_plots), sharex=True)
            axes_ep = [axes_ep] if len(self._episodic_plots) == 1 else axes_ep
            self._episodic_plot_figure.subplots_adjust(wspace=0.0, hspace=0.02)
            self._episodic_plot_figure.canvas.manager.set_window_title('Episodic Plots')
            axes_ep[-1].set_xlabel('Episode No')
            self._figures.append(self._episodic_plot_figure)
            for plot, axis in zip(self._episodic_plots, axes_ep):
                plot.initialize(axis)

        if len(self._step_plots) > 0:
            self._step_plot_figure, axes_int = plt.subplots(len(self._step_plots), sharex=True)
            axes_int = [axes_int] if len(self._step_plots) == 1 else axes_int
            self._step_plot_figure.canvas.manager.set_window_title('Step Plots')
            self._step_plot_figure.subplots_adjust(wspace=0.0, hspace=0.02)
            axes_int[-1].set_xlabel('Cumulative Steps')
            self._figures.append(self._step_plot_figure)
            for plot, axis in zip(self._step_plots, axes_int):
                plot.initialize(axis)

        if len(self._time_plots) > 0:
            self._time_plot_figure, axes_step = plt.subplots(len(self._time_plots), sharex=True)
            self._time_plot_figure.canvas.manager.set_window_title('Time Plots')
            axes_step = [axes_step] if len(self._time_plots) == 1 else axes_step
            self._time_plot_figure.subplots_adjust(wspace=0.0, hspace=0.2)
            axes_step[-1].set_xlabel('$t$/s')
            self._figures.append(self._time_plot_figure)
            for plot, axis in zip(self._time_plots, axes_step):
                plot.initialize(axis)

    def _update(self):
        """Called every *update cycle* steps to refresh the figure."""
        for plot in self._plots:
            plot.render()
        for fig in self._figures:
            fig.canvas.draw()
            fig.canvas.flush_events()
