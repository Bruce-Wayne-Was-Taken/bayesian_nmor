"""High-level driver for joint Bayesian estimation and posterior-grid zooming."""

from dataclasses import dataclass
from math import ceil

import cupy as cp

from joint_likelihood import calculate_joint_likelihood_gpu
from joint_posterior import JointPosterior, JointZoomConfig


@dataclass
class MeasurementUpdate:
    """One replayable Bayesian measurement interval."""

    measurement: cp.ndarray
    times: cp.ndarray
    bias: float 


class JointEstimator:
    """Apply joint updates, retain replay records, and refine posterior support."""

    def __init__(
        self,
        posterior,
        interpolator,
        params,
        zoom_config=None,
    ):
        self.posterior = posterior
        self.interpolator = interpolator
        self.params = params
        self.zoom_config = zoom_config
        self._validate_zoom_config(zoom_config)

        self.history = []
        self.update_records = []
        self.zoom_events = []
        self.support_history = []
        self._record_support_state()

    @staticmethod
    def _validate_zoom_config(config):
        if config is None:
            return
        if not isinstance(config, JointZoomConfig):
            raise TypeError("zoom_config must be a JointZoomConfig or None")
        if not 0.0 < config.credible_mass < 1.0:
            raise ValueError("credible_mass must lie in (0, 1)")
        if not 0.0 < config.trigger_span_ratio < 1.0:
            raise ValueError("trigger_span_ratio must lie in (0, 1)")
        if config.margin_cells < 0:
            raise ValueError("margin_cells must be non-negative")
        if config.growth_factor < 1.0:
            raise ValueError("growth_factor must be at least 1")
        if (
            config.max_grid_points_per_axis is not None
            and config.max_grid_points_per_axis < 2
        ):
            raise ValueError("max_grid_points_per_axis must be at least 2")

    def _record_support_state(self):
        self.support_history.append(
            {
                "grid_shape": (self.posterior.Nz, self.posterior.Ny),
                "bz_axis": self.posterior.bz_axis.get().copy(),
                "by_axis": self.posterior.by_axis.get().copy(),
            }
        )

    def _make_record(self, measurement, times, bias):
        return MeasurementUpdate(
            measurement=cp.array(measurement, dtype=cp.float64, copy=True),
            times=cp.array(times, dtype=cp.float64, copy=True),
            bias=float(cp.asarray(bias).get()),
        )

    #The function that does the posterior update no matter whether zoomed or not
    def _apply_record(self, record):
        log_likelihood = calculate_joint_likelihood_gpu(
            measurement=record.measurement,
            t_pts=record.times,
            bias=record.bias,
            bz_grid=self.posterior.bz_axis,
            by_grid=self.posterior.by_axis,
            interpolator=self.interpolator,
            sigma_noise=self.params.sigma_noise_longitudinal,
            likelihood_mode=self.params.likelihood_mode_longitudinal,
        )
        self.posterior.update(log_likelihood)

    @staticmethod
    def _next_axis_count(current_count, config):
        if config.max_grid_points_per_axis is None:
            return current_count
        return min(
            config.max_grid_points_per_axis,
            max(current_count, ceil(current_count * config.growth_factor)),
        )

    def _propose_axis(self, axis, hpd_lower, hpd_upper, physical_bounds):
        """Return ``(axis, zoomed)`` for one support dimension."""
        config = self.zoom_config
        current_lower = float(axis[0].get())
        current_upper = float(axis[-1].get())
        current_span = current_upper - current_lower
        hpd_span = hpd_upper - hpd_lower

        if hpd_span >= config.trigger_span_ratio * current_span:
            return axis.copy(), False

        point_count = self._next_axis_count(len(axis), config) #sets the number of points in the new axis based on the growth factor and max grid points per axis
        if point_count <= 2 * config.margin_cells + 1:
            raise ValueError("grid is too small for the configured zoom margin")

        # A one-cell HPD region still needs a finite interior width.
        current_spacing = current_span / (len(axis) - 1)
        interior_span = max(hpd_span, current_spacing) 
        spacing = interior_span / (point_count - 1 - 2 * config.margin_cells) 
        # the above two lines adds safety padding to the HPD just to be safe. Is this right? 
        lower = max(physical_bounds[0], hpd_lower - config.margin_cells * spacing)
        upper = min(physical_bounds[1], hpd_upper + config.margin_cells * spacing)

        if upper <= lower:
            return axis.copy(), False
        if upper - lower >= current_span * (1.0 - 1e-12):
            return axis.copy(), False

        return cp.linspace(lower, upper, point_count, dtype=cp.float64), True
    

    def maybe_zoom(self):
        """Refine contracted axes and replay every update on the new support.

        Returns a zoom-event dictionary, or ``None`` when the current support is
        still sufficiently broad.
        """
        if self.zoom_config is None:
            return None

        hpd_bounds = self.posterior.hpd_bounds(self.zoom_config.credible_mass)
        old_shape = (self.posterior.Nz, self.posterior.Ny)
        old_bz_bounds = (
            float(self.posterior.bz_axis[0].get()),
            float(self.posterior.bz_axis[-1].get()),
        )
        old_by_bounds = (
            float(self.posterior.by_axis[0].get()),
            float(self.posterior.by_axis[-1].get()),
        )

        bz_axis, zoomed_bz = self._propose_axis(
            self.posterior.bz_axis,
            hpd_bounds[0],
            hpd_bounds[1],
            self.posterior.physical_bz_bounds,
        )
        by_axis, zoomed_by = self._propose_axis(
            self.posterior.by_axis,
            hpd_bounds[2],
            hpd_bounds[3],
            self.posterior.physical_by_bounds,
        )
        if not (zoomed_bz or zoomed_by):
            return None

        self.posterior = JointPosterior(
            bz_axis,
            by_axis,
            physical_bz_bounds=self.posterior.physical_bz_bounds,
            physical_by_bounds=self.posterior.physical_by_bounds,
        )

        #how does the posterior handle the new data? FIXME 

        for record in self.update_records:
            self._apply_record(record)

        event = {
            "update_index": len(self.update_records),
            "old_shape": old_shape,
            "new_shape": (self.posterior.Nz, self.posterior.Ny),
            "old_bz_bounds": old_bz_bounds,
            "old_by_bounds": old_by_bounds,
            "new_bz_bounds": (
                float(self.posterior.bz_axis[0].get()),
                float(self.posterior.bz_axis[-1].get()),
            ),
            "new_by_bounds": (
                float(self.posterior.by_axis[0].get()),
                float(self.posterior.by_axis[-1].get()),
            ),
            "hpd_bounds": hpd_bounds,
            "zoomed_bz": zoomed_bz,
            "zoomed_by": zoomed_by,
        }
        self.zoom_events.append(event)
        return event

    def update(self, measurement, times, bias):
        """Apply one measurement update, then zoom and replay if required."""
        record = self._make_record(measurement, times, bias)
        self.update_records.append(record)
        self._apply_record(record)
        self.maybe_zoom()

        #does this reapply every posterior update for the zoomed posterior, isnt that insanely computationally expensive? What if we interpolate instead as well? FIXME

        summary = self.posterior.summary()
        self.history.append(summary)
        self._record_support_state()
        return summary

    def latest(self):
        """Return the most recent posterior summary, if one exists."""
        if not self.history:
            return None
        return self.history[-1]

    def reset_history(self):
        """Clear recorded summaries and support snapshots without changing state."""
        self.history = []
        self.zoom_events = []
        self.support_history = []
        self._record_support_state()
