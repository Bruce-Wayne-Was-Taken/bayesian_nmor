"""
joint_posterior.py

Joint Bayesian posterior over (Bz, By).

This class contains NO experiment logic,
NO interpolation,
NO likelihood calculation.

It simply stores and manipulates

        P(Bz, By)

on a rectangular support grid.
"""

from dataclasses import dataclass

import cupy as cp


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def logsumexp_gpu(logw):
    """
    Stable log(sum(exp(logw))).
    """
    m = cp.max(logw)
    return m + cp.log(cp.sum(cp.exp(logw - m)))


def normalize_log_weights(log_weights):
    """
    Normalize log-weights over the last two dimensions.

    Works for

        (Nz, Ny)

    and also

        (..., Nz, Ny)

    so it can normalize one posterior or thousands
    of temporary posteriors simultaneously.
    """

    max_log = cp.max(
        log_weights,
        axis=(-2, -1),
        keepdims=True,
    )

    shifted = log_weights - max_log

    log_norm = (
        max_log
        + cp.log(
            cp.sum(
                cp.exp(shifted),
                axis=(-2, -1),
                keepdims=True,
            )
        )
    )

    return log_weights - log_norm


# ----------------------------------------------------------------------
# Summary object
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


# ----------------------------------------------------------------------
# Posterior
# ----------------------------------------------------------------------

class JointPosterior:

    def __init__(self,
                 bz_axis,
                 by_axis):

        self.bz_axis = cp.asarray(bz_axis, dtype=cp.float64)
        self.by_axis = cp.asarray(by_axis, dtype=cp.float64)

        self.Nz = len(self.bz_axis)
        self.Ny = len(self.by_axis)

        # Uniform prior in log-space
        self.log_weights = cp.zeros(
            (self.Nz, self.Ny),
            dtype=cp.float64
        )

        self.normalize()

    # -------------------------------------------------------------

    @property
    def weights(self):
        return cp.exp(self.log_weights)

    # -------------------------------------------------------------

    def normalize(self):

        self.log_weights = normalize_log_weights(
            self.log_weights
    )

    # -------------------------------------------------------------

    def update(self,
               log_likelihood):
        """
        Bayesian update.

        Parameters
        ----------
        log_likelihood

        shape

            (Nz,Ny)
        """

        if log_likelihood.shape != self.log_weights.shape:
            raise ValueError(
                f"Likelihood shape {log_likelihood.shape}"
                f" != posterior shape {self.log_weights.shape}"
            )

        self.log_weights = self.log_weights + log_likelihood

        self.normalize()

    # -------------------------------------------------------------

    def MAP(self):

        idx = cp.argmax(self.log_weights)

        iy, iz = cp.unravel_index(idx,
                                  self.log_weights.shape)

        return (
            float(self.bz_axis[iz].get()),
            float(self.by_axis[iy].get())
        )

    # -------------------------------------------------------------

    def mean(self):

        w = self.weights

        ZZ, YY = cp.meshgrid(
            self.bz_axis,
            self.by_axis,
            indexing="xy"
        )

        mean_bz = cp.sum(w * ZZ)
        mean_by = cp.sum(w * YY)

        return (
            float(mean_bz.get()),
            float(mean_by.get())
        )

    # -------------------------------------------------------------

    def covariance(self):

        w = self.weights

        ZZ, YY = cp.meshgrid(
            self.bz_axis,
            self.by_axis,
            indexing="xy"
        )

        mean_bz = cp.sum(w * ZZ)
        mean_by = cp.sum(w * YY)

        dz = ZZ - mean_bz
        dy = YY - mean_by

        var_z = cp.sum(w * dz * dz)

        var_y = cp.sum(w * dy * dy)

        cov = cp.sum(w * dz * dy)

        return cp.array(
            [
                [var_z, cov],
                [cov, var_y]
            ]
        )

    # -------------------------------------------------------------

    def entropy(self):

        w = self.weights

        eps = 1e-300

        return float(
            (-cp.sum(w * cp.log(w + eps))).get()
        )

    # -------------------------------------------------------------

    def marginal_bz(self):

        """
        P(Bz)
        """

        return cp.sum(
            self.weights,
            axis=1
        )

    # -------------------------------------------------------------

    def marginal_by(self):

        """
        P(By)
        """

        return cp.sum(
            self.weights,
            axis=0
        )

    # -------------------------------------------------------------

    def credible_region(self,
                        mass=0.95):
        """
        Returns the posterior threshold
        enclosing the requested probability.

        Useful for adaptive zoom.
        """

        if not (0.0 < mass < 1.0):
            raise ValueError("mass must lie in (0,1)")

        flat = cp.sort(
            self.weights.ravel()
        )[::-1]

        cumulative = cp.cumsum(flat)

        idx = int(
            cp.searchsorted(
                cumulative,
                mass
            )
        )

        return float(flat[idx].get())

    # -------------------------------------------------------------

    def summary(self):

        cov = self.covariance()

        std_bz = float(cp.sqrt(cov[0, 0]).get())

        std_by = float(cp.sqrt(cov[1, 1]).get())

        #we have calculated the covariance as well, return and save that as well FIXME

        mean_bz, mean_by = self.mean()

        map_bz, map_by = self.MAP()

        return PosteriorSummary(
            map_bz=map_bz,
            map_by=map_by,

            mean_bz=mean_bz,
            mean_by=mean_by,

            std_bz=std_bz,
            std_by=std_by,

            covariance=cov,

            entropy=self.entropy()
        )
    
    def sample(
        self,
        n_samples,
        return_indices=False,
    ):
        """
        Draw samples from the discrete joint posterior.

        Returns
        -------
        bz : (n_samples,)
        by : (n_samples,)

        optionally

        iz : Bz grid indices
        iy : By grid indices
        """

        flat = self.weights.ravel()

        flat_indices = cp.random.choice(
            flat.size,
            size=n_samples,
            p=flat,
        )

        iz, iy = cp.unravel_index(
            flat_indices,
            self.weights.shape,
        )

        bz = self.bz_axis[iz]
        by = self.by_axis[iy]

        if return_indices:
            return bz, by, iz, iy

        return bz, by