from gym_electric_motor.core import ElectricMotorVisualization
from .motor_dashboard_plots import StatePlot, ActionPlot, RewardPlot, StepPlot, EpisodicPlot, IntervalPlot
import matplotlib.pyplot as plt
import gym


class MotorDashboard(ElectricMotorVisualization):
    """A dashboard to plot the GEM states into graphs.

    Every MotorDashboard consists of multiple MotorDashboardPlots that are each responsible for the plots in a single
    matplotlib axis.

    It handles three different types of plots: The StepPlot, EpisodicPlot and IntervalPlot which especially differ in
    their x-Axis. The step plots plot every step and have got the time on the x-Axis. The EpisodicPlots plot statistics
    over the episodes (e.g. mean reward per step in each episode). The episode number is on their x-Axis. The
    IntervalPlots plot statistics over the last taken steps (e.g. mean reward over the last 1000 steps) and their x-Axis
    are the cumulative number of steps.

    The StepPlots, EpisodicPlots and IntervalPlots each are plotted into three separate figures.

    The most common StepPlots (i.e to plot the states, actions and rewards) can be plotted by just passing the
    corresponding strings in the constructor. Additional plots (e.g. the MeanEpisodeRewardPlot) have to be instantiated
    manually and passed to the constructor.

    Furthermore, completely custom plots can be defined that have to derive from the StatePlot, EpisodicPlot or
    IntervalPlot base classes.
    """

    def __init__(self, state_plots=(), action_plots=(), reward_plot=False, step_plots=(),
                 episodic_plots=(), interval_plots=(), update_interval=1000, step_plot_width=10000, style=None, **__):
        """
        Args:
            state_plots('all'/iterable(str)): An iterable of state names to be shown. If 'all' all states will be shown.
                Default: () (no plotted states)
            action_plots('all'/iterable(int)): If action_plots='all', all actions will be plotted. If more than one
                action can be applied on the environment it can be selected by its index.
                Default: () (no plotted actions).
            reward_plot(boolean): Select if the current reward is to be plotted. Default: False
            step_plots(iterable(StepPlot)): Additional custom step plots. Default: ()
            episodic_plots(iterable(EpisodicPlot)): Additional already instantiated EpisodePlots to be shown
            interval_plots(iterable(IntervalPlot)): Additional already instantiated IntervalPlots to be shown
            update_interval(int > 0): Amount of steps after which the plots are updated. Updating each step reduces the
                performance drastically. Default: 1000
            step_plot_width(int > 0): Width of the step plots in steps. Default: 10000 steps
                (1 second for continuously controlled environments / 0.1 second for discretely controlled environments)
            style(string): Select one of the matplotlib-styles. e.g. "dark-background".
                Default: None (the already selected style)
        """
        assert type(reward_plot) is bool
        assert all(isinstance(sp, StepPlot) for sp in step_plots)
        assert all(isinstance(ep, EpisodicPlot) for ep in episodic_plots)
        assert all(isinstance(ip, IntervalPlot) for ip in interval_plots)
        assert type(update_interval) in [int, float]
        assert update_interval > 0
        assert type(step_plot_width) in [int, float]
        assert step_plot_width > 0
        assert style in plt.style.available or style is None

        super().__init__()
        if style is not None:
            plt.style.use(style)
        self._figures = []
        self._step_plot_figure = None
        self._episodic_plot_figure = None
        self._interval_plot_figure = None
        self._reset = False

        self._state_plots = state_plots
        self._action_plots = action_plots
        self._reward_plot = reward_plot

        self._custom_step_plots = list(step_plots)
        self._step_plots = []
        self._episodic_plots = list(episodic_plots)
        self._interval_plots = list(interval_plots)
        self._update_interval = int(update_interval)
        self._step_plot_width = int(step_plot_width)
        self._plots = []
        self._k = 0

    def on_reset_begin(self):
        """Called before the environment is reset. All subplots are reset.
        """
        for plot in self._plots:
            plot.on_reset_begin()
        self._reset = True

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
            reward(float): The reward that has been received for the last action on the last state :math:`r_{k-1}`.
            done(bool): Flag, that indicates, if the last action lead to a terminal state :math:`t_{k-1}`.
        """
        for plot in self._plots:
            plot.on_step_end(k, state, reference, reward, done)

    def render(self):
        """Updates the plots every _update cycle_ calls of this method."""
        if not (self._step_plot_figure or self._episodic_plot_figure or self._interval_plot_figure) \
           and len(self._plots) > 0:
            self._initialize()
        self._k += 1
        if self._k % self._update_interval == 0:
            self._update()

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
            elif type(env.action_space) is gym.spaces.Box:
                self._action_plots = list(range(env.action_space.shape[0]))
        self._step_plots = []

        if len(self._state_plots) > 0:
            assert all(state in state_names for state in self._state_plots)
            for state in self._state_plots:
                self._step_plots.append(StatePlot(state))

        if len(self._action_plots) > 0:
            assert type(env.action_space) in (gym.spaces.Box, gym.spaces.Discrete), \
                f'Action space of type {type(env.action_space)} not supported for plotting.'
            for action in self._action_plots:
                ap = ActionPlot(action)
                self._step_plots.append(ap)

        if self._reward_plot:
            self._reward_plot = RewardPlot()
            self._state_plots.append(self._reward_plot)
            self._step_plots.append(self._reward_plot)

        self._plots = self._step_plots + self._episodic_plots + self._interval_plots

        for step_plot in self._step_plots:
            step_plot.set_width(self._step_plot_width)

        for plot in self._plots:
            plot.set_env(env)

    def _initialize(self):
        """Called with first render() call to setup the figures and plots.
        """
        plt.close()
        self._figures = []

        # create separate figures for time based, interval and episode based plots
        if len(self._step_plots) > 0:
            self._step_plot_figure, axes_step = plt.subplots(len(self._step_plots), sharex=True)
            axes_step = [axes_step] if len(self._step_plots) == 1 else axes_step
            self._step_plot_figure.subplots_adjust(wspace=0.0, hspace=0.2)
            axes_step[-1].set_xlabel('$t$/s')
            self._figures.append(self._step_plot_figure)
            for plot, axis in zip(self._step_plots, axes_step):
                plot.initialize(axis)

        if len(self._episodic_plots) > 0:
            self._episodic_plot_figure, axes_ep = plt.subplots(len(self._episodic_plots))
            axes_ep = [axes_ep] if len(self._episodic_plots) == 1 else axes_ep
            self._episodic_plot_figure.subplots_adjust(wspace=0.0, hspace=0.02)
            axes_ep[-1].set_xlabel('Episode No')
            self._figures.append(self._episodic_plot_figure)
            for plot, axis in zip(self._episodic_plots, axes_ep):
                plot.initialize(axis)

        if len(self._interval_plots) > 0:
            self._interval_plot_figure, axes_int = plt.subplots(len(self._interval_plots))
            axes_int = [axes_int] if len(self._interval_plots) == 1 else axes_int
            self._interval_plot_figure.subplots_adjust(wspace=0.0, hspace=0.02)
            axes_int[-1].set_xlabel('Cumulative Steps')
            for plot, axis in zip(self._interval_plots, axes_int):
                plot.initialize(axis)

        plt.pause(0.1)

    def _update(self):
        """Called every *update cycle* steps to refresh the figure.
        """
        for plot in self._plots:
            plot.render()
        for fig in self._figures:
            fig.canvas.draw()
            fig.canvas.flush_events()
