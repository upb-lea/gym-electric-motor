from gym_electric_motor.visualization.motor_dashboard_plots.base_plots import TimePlot
import numpy as np


class ExternalPlot(TimePlot):
    """
        Class to plot lines that do not belong to the state of the environment. A reference and any number of additional
        lines can be plotted.
        Usage Example
        -------------
            if __name__ == '__main__':
                external_plot = ExternalPlot(min=-1, max=1, reference=True, additional_lines=2)

                env = gem.make('DqCont-CC-PMSM-v0', visualization=MotorDashboard(state_plots=['i_sd', 'i_sq'],
                                                                                additional_plots=(external_plot,)))

                external_plot.set_label({'y_label': 'y', 'state_label': '$state$', 'ref_label': '$reference$',
                                        'add_label': ['$add_1$', '$add_2$']})
                done = True
                for t in range(100000):
                if done:
                    state, reference = env.reset()
                env.render()
                data = [np.sin(t / 500), np.sin(t / 1000), np.sin(t / 1500), np.sin(t / 2000)]
                external_plot.add_data(data)
                (state, reference), reward, done, _ = env.step([0, 0])
    """

    def __init__(self, referenced=False, additional_lines=0, min=0, max=1):
        super().__init__()

        self._state_line_config = self._default_time_line_cfg.copy()
        self._ref_line_config = self._default_time_line_cfg.copy()
        self._add_line_config = self._default_time_line_cfg.copy()

        self._referenced = referenced
        self.min = min
        self.max = max

        self._state_line = None
        self._reference_line = None

        self._state_data = []
        self._reference_data = []
        self._additional_data = []

        self.added = additional_lines > 0
        self.add_lines = additional_lines

        self.state_label = ''
        self.ref_label = ''

        if self.added:
            self.add_labels = []
            self._additional_lines = []
            for i in range(additional_lines):
                self._additional_lines.append([])
                self._additional_data.append(None)
                self.add_labels.append('')

    def set_env(self, env):
        super().set_env(env)
        self._label = None
        self._y_lim = (self.min, self.max)
        self.reset_data()

    def reset_data(self):
        super().reset_data()
        self._state_data = np.full(shape=self._x_data.shape, fill_value=np.nan)
        self._reference_data = np.full(shape=self._x_data.shape, fill_value=np.nan)

        if self.added:
            for i in range(self.add_lines):
                self._additional_data[i] = np.full(shape=self._x_data.shape, fill_value=np.nan)

    def initialize(self, axis):
        # Docstring of superclass
        super().initialize(axis)

        self._state_line, = self._axis.plot(self._x_data, self._state_data, **self._state_line_config, zorder=self.add_lines+2)
        self._lines = [self._state_line]

        if self._referenced:
            self._reference_line, = self._axis.plot(self._x_data, self._reference_data, **self._ref_line_config, zorder=self.add_lines+1)
            axis.lines = axis.lines[::-1]
            self._lines.append(self._reference_line)

        self._y_data = [self._state_data, self._reference_data]

        if self.added:
            for i in range(self.add_lines):
                self._additional_lines[i], = self._axis.plot(self._x_data, self._additional_data[i], **self._add_line_config, zorder=self.add_lines-i)
                self._lines.append(self._additional_lines[i])
                self._y_data.append(self._additional_data[i])

        if self._referenced:
            if self.added:
                lines = [self._state_line, self._reference_line]
                lines.extend(self._additional_lines)
                labels = [self.state_label, self.ref_label]
                labels.extend(self.add_labels)
                self._axis.legend((lines), (labels), loc='upper left', numpoints=20)
            else:
                self._axis.legend(([self._state_line, self._reference_line]), ([self.state_label, self.ref_label]), loc='upper left', numpoints=20)
        else:
            self._axis.legend((self._state_line, ), (self.state_label, ), loc='upper left', numpoints=20)

    def set_label(self, labels):
        """Method to set the labels"""
        self._label = labels.get('y_label', '')
        self.state_label = labels['state_label']
        if self._referenced:
            self.ref_label = labels.get('ref_label', '')
        if 'add_label' in labels.keys():
            self.add_labels = labels['add_label']

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        idx = self.data_idx
        self._x_data[idx] = self._t

    def add_data(self, additional_data):
        """Method to pass the external data"""
        idx = self.data_idx
        if self._referenced:
            self._state_data[idx] = additional_data[0]
            self._reference_data[idx] = additional_data[1]
            if self.added:
                for i in range(self.add_lines):
                    self._additional_data[i][idx] = additional_data[i + 2]
        elif self.added:
            self._state_data[idx] = additional_data[0]
            for i in range(self.add_lines):
                self._additional_data[i][idx] = additional_data[i + 1]
        else:
            self._state_data[idx] = additional_data[0]
