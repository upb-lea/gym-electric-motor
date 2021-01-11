from .base_plots import StepPlot


class CumulativeConstraintViolationPlot(StepPlot):
    """Plots the cumulative number of constraint violations during the runtime of the env. into an axis."""

    def __init__(self):
        super().__init__()
        self._no_of_violations = 0
        self._violations = [0]
        self._label = 'Cum. No. of Constraint Violations'
        self._x_data.append(0)

    def initialize(self, axis):
        super().initialize(axis)
        self._y_data.append(self._violations)
        self._lines.append(self._axis.plot(self._x_data, self._violations)[0])

    def reset_data(self):
        super().reset_data()
        self._violations = [0]

    def on_step_end(self, k, state, reference, reward, done):
        super().on_step_end(k, state, reference, reward, done)
        if done:
            # Add another point for a step-like plot
            self._x_data.append(self._k - 1)
            self._violations.append(self._no_of_violations)
            self._no_of_violations += 1
            self._x_data.append(self._k)
            self._violations.append(self._no_of_violations)

    def _scale_y_axis(self):
        # Read the limit before writing, because reading is fast, but writing is slow
        if self._axis.get_xlim() != (-1, self._no_of_violations + 1):
            self._axis.set_ylim(-1, self._no_of_violations + 1)
