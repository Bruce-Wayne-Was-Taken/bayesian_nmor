"""
joint_estimator.py

High-level driver for joint Bayesian estimation.

Responsibilities
----------------
1. Compute the joint likelihood
2. Update the posterior
3. Store estimation history

No KL.
No adaptive zoom.
No replay loop.

Those will be added later.
"""

from joint_likelihood import calculate_joint_likelihood_gpu


class JointEstimator:

    def __init__(
        self,
        posterior,
        interpolator,
        params,
    ):
        self.posterior = posterior
        self.interpolator = interpolator
        self.params = params

        self.history = []

    # ----------------------------------------------------------

    def update(
        self,
        measurement,
        times,
        bias,
    ):
        """
        Perform one Bayesian update.

        Parameters
        ----------
        measurement : (Nt,)
            Experimental signal.

        times : (Nt,)
            Time axis.

        bias : float
            Applied longitudinal bias.

        Returns
        -------
        PosteriorSummary
        """

        logL = calculate_joint_likelihood_gpu(
            measurement=measurement,
            t_pts=times,
            bias=bias,
            bz_grid=self.posterior.bz_axis,
            by_grid=self.posterior.by_axis,
            interpolator=self.interpolator,
            sigma_noise=self.params.sigma_noise_longitudinal,
            likelihood_mode=self.params.likelihood_mode_longitudinal,
        )

        self.posterior.update(logL)

        summary = self.posterior.summary()

        self.history.append(summary)

        return summary

    # ----------------------------------------------------------

    def latest(self):
        """
        Return most recent summary.
        """

        if len(self.history) == 0:
            return None

        return self.history[-1]

    # ----------------------------------------------------------

    def reset_history(self):

        self.history = []