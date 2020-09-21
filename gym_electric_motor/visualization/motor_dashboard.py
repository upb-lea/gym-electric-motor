from gym_electric_motor.core import ElectricMotorVisualization
import matplotlib.pyplot as plt


class MotorDashboard(ElectricMotorVisualization):
    """Dashboard to plot the GEM states into graphs.

    Every MotorDashboard consists of multiple MotorDashboardPlots that are each responsible for the plots in a single
    matplotlib axis.
    The MotorDashboard is responsible for all matplotlib.figure related tasks, especially updating the figures.

    """

    def __init__(self, state_plots=(), action_plots=(), reward_plot=False, custom_plots=(), style=None, **__):
        """
        Args:
            state_plots(iterable): A list of plots to show. Each element may either be a string or an already instantiated
                MotorDashboardPlot
                Possible strings:
                    - {state_name}: The corresponding state is plotted
                    - reward: The reward per step is plotted
                    - action_{i}: The i-th action is plotted. 'action_0' for discrete action space
                    - mean_reward: The mean episode reward
                    
            update_cycle(int): Number after how many steps the plot shall be updated. (default 1000)
            dark_mode(Bool):  Select a dark background for visualization by setting it to True
        """
        assert len(state_plots) > 0, "plots needs to be an iterable of len > 0"
        super().__init__()
        plt.style.use(style)
        self._step_plot_figure = None
        self._episode_plot_figure = None
        self._interval_plot_figure = None

        self._step_plots = []
        self._interval_plots = []
        self._episodic_plots = []

        self._reset = False
        self._plots = plots

    def on_reset_begin(self):
        """Called when the environment is reset. All subplots are reset.
        """

        for plot in self._plots:
            plot.on_reset_begin()
        for plot in self._episodic_plots:
            plot.on_reset_begin()
        self._reset = True

    def on_reset_end(self, observation):
        """Called when the environment is reset. All subplots are reset.
        """

        for plot in self._plots:
            plot.on_reset_end(observation)
        for plot in self._episodic_plots:
            plot.on_reset_end(observation)

    def on_step_begin(self, action):
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
        for plot in self._plots:
            plot.on_step_begin(action)
        for plot in self._episodic_plots:
            plot.on_step_begin(action)

    def on_step_end(self, observation, reward, done):
        for plot in self._plots:
            plot.on_step_end(observation, reward, done)
        for plot in self._episodic_plots:
            plot.on_step_end(observation, reward, done)

    def render(self):
        if not (self._step_plot_figure or self._episode_plot_figure):
            self._initialize()

        if (self._k + 1) % self._update_cycle == 0:
            self._update()

    def set_env(self, env):
        """Called during initialization of the environment to interconnect all modules. State_names, references,...
        might be saved here for later processing

        Args:
            ps(PhysicalSystem): PhysicalSystem of the environment
            rg(ReferenceGenerator): ReferenceGenerator of the environment
            rf(RewardFunction): RewardFunction of the environment
        """
        for plot in self._plots:
            plot.set_env(env)
        for plot in self._episodic_plots:
            plot.set_env(env)

    def _initialize(self):
        """Called with first render() call to setup the figures and plots.
        """
        axis_ep = []
        plt.close()

        # create separate figures for time based and episode based plots
        self._step_plot_figure, axes = plt.subplots(len(self._plots), sharex=True)
        if self._episodic_plots:
            self._episode_plot_figure, axes_ep = plt.subplots(len(self._episodic_plots))
            self._episode_plot_figure.subplots_adjust(wspace=0.0, hspace=0.02)
            self._step_plot_figure.subplots_adjust(wspace=0.0, hspace=0.2)
            self._episode_plot_figure.text(0.5, 0.04, 'episode', va='center', ha='center')

        # adding a common x-label to all the subplots in each figure
        self._step_plot_figure.text(0.5, 0.04, 't/s', va='center', ha='center')

        # plt.subplot() does not return an iterable var when the number of subplots==1
        if len(self._plots) == 1:
            axes = [axes]
        if len(self._episodic_plots) == 1:
            axis_ep = [axes_ep]
        for plot, axis in zip(self._plots, axes):
            plot.initialize(axis)

        for plot, axis in zip(self._episodic_plots, axis_ep):
            plot.initialize(axis)
        plt.pause(0.1)

    def _update(self):
        """Called every {update_cycle} steps to refresh the figure.
        """
        for plot in self._plots:
            plot.update()
        if self._reset:
            self._episode_plot_figure.draw()
            self._episode_plot_figure.flush_events()
            self._reset = False
        self._step_plot_figure.canvas.draw()
        self._step_plot_figure.canvas.flush_events()
