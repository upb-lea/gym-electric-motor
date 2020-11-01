from .base_plots import StepPlot


class CumulativeConstraintViolationPlot(StepPlot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_of_violations = 0
        self._violations = [0]
        self._label = 'Cum. No. of Constraint Violations'
        self._t_data.append(0)

    def initialize(self, axis):
        super().initialize(axis)
        self._y_data.append(self._violations)
        self._lines.append(self._axis.plot(self._t_data, self._violations)[0])

    def on_step_end(self, k, state, reference, reward, done):
        if done:
            self._no_of_violations += 1

    def _set_y_data(self):
        self._violations.append(self._no_of_violations)

    def _scale_x_axis(self):
        self._axis.set_xlim(-1, self._t_data[-1] + 1)

    def _scale_y_axis(self):
        self._axis.set_ylim(-1, self._no_of_violations + 1)
