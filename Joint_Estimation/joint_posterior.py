"""
joint_posterior.py

Joint Bayesian posterior over (Bz, By) on a uniform rectangular support grid.
"""

from dataclasses import dataclass
from typing import Optional

import cupy as cp


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def logsumexp_gpu(logw):
    """Stable ``log(sum(exp(logw)))``."""
    maximum = cp.max(logw)
    return maximum + cp.log(cp.sum(cp.exp(logw - maximum)))


def normalize_log_weights(log_weights):
    """Normalize log-weights over the final ``(Nz, Ny)`` dimensions."""
    max_log = cp.max(log_weights, axis=(-2, -1), keepdims=True)
    shifted = log_weights - max_log
    log_norm = max_log + cp.log(
        cp.sum(cp.exp(shifted), axis=(-2, -1), keepdims=True)
    )
    return log_weights - log_norm


# ----------------------------------------------------------------------
# Data objects
# ----------------------------------------------------------------------

@dataclass
class PosteriorSummary:
    map_bz: float
    map_by: float
    mean_bz: float
    mean_by: float
    std_bz: float
    std_by: float
    covariance: cp.ndarray
    entropy: float


@dataclass(frozen=True)
class JointZoomConfig:
    """Settings for conservative uniform-grid posterior refinement."""

    credible_mass: float = 0.999
    trigger_span_ratio: float = 0.25 
    margin_cells: int = 8 #what should be the ideal value? FIXME
    growth_factor: float = 1.5
    max_grid_points_per_axis: Optional[int] = None


# ----------------------------------------------------------------------
# Posterior
# ----------------------------------------------------------------------

class JointPosterior:
    """Discrete posterior with array convention ``weights[iz, iy] = P(Bz, By)``."""

    def __init__(
        self,
        bz_axis,
        by_axis,
        physical_bz_bounds=None,
        physical_by_bounds=None,
    ):
        self.bz_axis = cp.asarray(bz_axis, dtype=cp.float64)
        self.by_axis = cp.asarray(by_axis, dtype=cp.float64)
        self._validate_axis(self.bz_axis, "bz_axis")
        self._validate_axis(self.by_axis, "by_axis")

        self.Nz = len(self.bz_axis)
        self.Ny = len(self.by_axis)
        self.physical_bz_bounds = self._resolve_physical_bounds(
            physical_bz_bounds,
            self.bz_axis,
            "physical_bz_bounds",
        )
        self.physical_by_bounds = self._resolve_physical_bounds(
            physical_by_bounds,
            self.by_axis,
            "physical_by_bounds",
        )

        # Uniform prior in log-space.
        self.log_weights = cp.ones((self.Nz, self.Ny), dtype=cp.float64)
        self.normalize()

    @staticmethod
    def _validate_axis(axis, name):
        if len(axis) < 2:
            raise ValueError(f"{name} must contain at least two points")
        if bool(cp.any(axis[1:] <= axis[:-1]).get()):
            raise ValueError(f"{name} must be strictly increasing")

    @staticmethod
    def _resolve_physical_bounds(bounds, axis, name):
        if bounds is None:
            return (float(axis[0].get()), float(axis[-1].get()))

        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} must be an increasing (lower, upper) pair")

        axis_lower = float(axis[0].get())
        axis_upper = float(axis[-1].get())
        if bounds[0] > axis_lower or bounds[1] < axis_upper:
            raise ValueError(f"{name} must contain the current support")

        return (float(bounds[0]), float(bounds[1]))

    @property
    def weights(self):
        return cp.exp(self.log_weights)

    def normalize(self):
        self.log_weights = normalize_log_weights(self.log_weights)

    def update(self, log_likelihood):
        """Apply a log-likelihood of shape ``(Nz, Ny)`` and normalize."""
        if log_likelihood.shape != self.log_weights.shape:
            raise ValueError(
                f"Likelihood shape {log_likelihood.shape} "
                f"!= posterior shape {self.log_weights.shape}"
            )
        self.log_weights = self.log_weights + log_likelihood
        self.normalize()

    def MAP(self):
        """Return the maximum-a-posteriori ``(Bz, By)`` support point."""
        index = cp.argmax(self.log_weights)
        iz, iy = cp.unravel_index(index, self.log_weights.shape)
        return (
            float(self.bz_axis[iz].get()),
            float(self.by_axis[iy].get()),
        )

    def _coordinate_mesh(self):
        return cp.meshgrid(self.bz_axis, self.by_axis, indexing="ij")

    def mean(self):
        """Return posterior mean ``(Bz, By)``."""
        weights = self.weights
        bz_mesh, by_mesh = self._coordinate_mesh()
        return (
            float(cp.sum(weights * bz_mesh).get()),
            float(cp.sum(weights * by_mesh).get()),
        )

    def covariance(self):
        """Return the 2-by-2 posterior covariance matrix for ``(Bz, By)``."""
        weights = self.weights
        bz_mesh, by_mesh = self._coordinate_mesh()
        mean_bz = cp.sum(weights * bz_mesh)
        mean_by = cp.sum(weights * by_mesh)
        delta_bz = bz_mesh - mean_bz
        delta_by = by_mesh - mean_by
        return cp.array(
            [
                [cp.sum(weights * delta_bz * delta_bz), cp.sum(weights * delta_bz * delta_by)],
                [cp.sum(weights * delta_bz * delta_by), cp.sum(weights * delta_by * delta_by)],
            ]
        )

    def entropy(self):
        weights = self.weights
        return float((-cp.sum(weights * cp.log(weights + 1e-300))).get())

    def marginal_bz(self):
        """Return ``P(Bz)`` with shape ``(Nz,)``."""
        return cp.sum(self.weights, axis=1)

    def marginal_by(self):
        """Return ``P(By)`` with shape ``(Ny,)``."""
        return cp.sum(self.weights, axis=0)

    def credible_region(self, mass=0.95):
        """Return the probability threshold enclosing at least ``mass``."""
        if not (0.0 < mass < 1.0):
            raise ValueError("mass must lie in (0, 1)")

        flat = cp.sort(self.weights.ravel())[::-1]
        cumulative = cp.cumsum(flat)
        index = min(int(cp.searchsorted(cumulative, cp.asarray(mass)).get()), len(flat) - 1)
        return float(flat[index].get())

    def hpd_bounds(self, mass=0.999):
        """Return axis-aligned bounds of the discrete highest-density region.

        Disconnected modes are intentionally enclosed by one rectangle so zooming
        never discards a mode merely because the support between modes is sparse.
        """
        threshold = self.credible_region(mass)
        iz, iy = cp.where(self.weights >= threshold)
        return (
            float(self.bz_axis[cp.min(iz)].get()),
            float(self.bz_axis[cp.max(iz)].get()),
            float(self.by_axis[cp.min(iy)].get()),
            float(self.by_axis[cp.max(iy)].get()),
        )

    def summary(self):
        covariance = self.covariance()
        mean_bz, mean_by = self.mean()
        map_bz, map_by = self.MAP()
        return PosteriorSummary(
            map_bz=map_bz,
            map_by=map_by,
            mean_bz=mean_bz,
            mean_by=mean_by,
            std_bz=float(cp.sqrt(covariance[0, 0]).get()),
            std_by=float(cp.sqrt(covariance[1, 1]).get()),
            covariance=covariance,
            entropy=self.entropy(),
        )

    def sample(self, n_samples, return_indices=False):
        """Draw samples from the discrete joint posterior."""
        flat_indices = cp.random.choice(
            self.weights.size,
            size=n_samples,
            p=self.weights.ravel(),
        )
        iz, iy = cp.unravel_index(flat_indices, self.weights.shape)
        bz = self.bz_axis[iz]
        by = self.by_axis[iy]
        if return_indices:
            return bz, by, iz, iy
        return bz, by
