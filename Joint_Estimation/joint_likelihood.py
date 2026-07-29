"""
Tensor conventions
------------------

Prediction : (Nt, Nz, Ny) # need new dimension after z for bias -  number of points decided on the set bias FIXME

Residual : (Nt, Nz, Ny) # need new dimension after z for bias FIXME
 
Likelihood : (Nz, Ny) # need new dimension after z for bias FIXME

Posterior : (Nz, Ny) # need new dimension after z for bias FIXME
"""
import cupy as cp


def get_predictions_joint(
    interpolator,
    t_pts,
    bz_grid,
    by_grid,
    bias,
):
    """
    Predict the full transient response over the joint parameter grid.

    Parameters
    ----------
    t_pts : (Nt,)
    bz_grid : (Nz,) # need new dimension after z for bias, and it will be (Nz, Nbias) FIXME
    by_grid : (Ny,)
    bias : scalar

    Returns
    -------
    prediction : (Nt, Nz, Ny) # need new dimension after z for bias, (Nt, Nz, Nbias, Ny) FIXME
    """

    Nt = len(t_pts)
    Nz = len(bz_grid)
    # Nbias, how are we going to query this grid? bias set points, plus minus least count, or do we consider more values in between?
    Ny = len(by_grid)

    # ---------------------------------------------------------
    # Broadcast everything
    # ---------------------------------------------------------

    T = t_pts[:, None, None] # need new dimension after z for bias FIXME

    Z = bz_grid[None, :, None] + bias # need new dimension after z for bias FIXME

    Y = by_grid[None, None, :] # need new dimension after z for bias FIXME

    final_shape = (Nt, Nz, Ny) # need new dimension after z for bias, (Nt, Nz, Nbias, Ny) FIXME

    T = cp.broadcast_to(T, final_shape)

    Z = cp.broadcast_to(Z, final_shape)

    # need new dimension after z for bias, need a broadcaster for bias FIXME

    Y = cp.broadcast_to(Y, final_shape)

    prediction = interpolator.interpolate(
        T.ravel(),
        Z.ravel(),
        Y.ravel(),
    )

    return prediction.reshape(final_shape)


def calculate_joint_likelihood_gpu(
    measurement,
    t_pts,
    bias, # need new dimension after z for bias FIXME
    bz_grid,
    by_grid,
    interpolator,
    sigma_noise,
    likelihood_mode="Gaussian",
):
    """
    Joint likelihood over (Bz, By).

    Returns
    -------
    logL : (Nz, Ny) # need new dimension after z for bias, and do a simple marginalisation average over the bias FIXME
    """

    prediction = get_predictions_joint(
        interpolator,
        t_pts,
        bz_grid,
        by_grid,
        bias,
    )

    if likelihood_mode == "Gaussian":

        residual = (
            measurement[:, None, None]
            - prediction
        )

        SSE = cp.sum(
            residual * residual,
            axis=0,
        )

        logL = -SSE / (2 * sigma_noise**2)

    elif likelihood_mode == "Cauchy":

        gamma = sigma_noise * 1.177

        residual = (
            measurement[:, None, None]
            - prediction
        )

        log_term = cp.log(
            1
            + residual * residual / gamma**2
        )

        logL = -cp.sum(
            log_term,
            axis=0,
        )

    else:

        raise ValueError(
            f"Unknown likelihood mode {likelihood_mode}"
        )

    # numerical stability

    logL = logL - cp.max(logL)
    del residual
    return logL