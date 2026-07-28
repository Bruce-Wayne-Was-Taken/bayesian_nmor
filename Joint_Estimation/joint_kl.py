"""
joint_kl.py

Monte Carlo Expected Information Gain (KL Utility)

This module performs Bayesian experimental design
using the current joint posterior.

Tensor conventions
------------------

Prediction

    (Nb,Np,Nt)

Likelihood

    (Nb,Np,Nz,Ny)

Posterior

    (Nb,Np,Nz,Ny)

KL

    (Nb,Np)

Utility

    (Nb,)
"""

import cupy as cp

from joint_posterior import normalize_log_weights
from joint_likelihood import get_predictions_joint

def compute_kl_gpu(
    prior_log,
    posterior_log,
):
    """
    Computes

        KL(posterior || prior)

    Parameters
    ----------

    prior_log

        (...,Nz,Ny)

    posterior_log

        (...,Nz,Ny)

    Returns
    -------

    KL

        (...)
    """

    posterior = cp.exp(posterior_log)

    kl = posterior * (
        posterior_log
        - prior_log
    )

    return cp.sum(
        kl,
        axis=(-2, -1),
    )

def generate_synthetic_observations_gpu(
    predictions,
    sigma_noise,
):
    """
    Parameters
    ----------

    predictions

        (Nb,Np,Nt)

    Returns
    -------

    observations

        (Nb,Np,Nt)
    """

    noise = cp.random.normal(
        loc=0.0,
        scale=sigma_noise,
        size=predictions.shape,
    )

    return predictions + noise


#how do i condition on just one point?
def build_prediction_times(
    current_time,
    next_time,
    params,
):
    """
    Returns the time grid used for
    utility evaluation.
    """

    if params.kl_time_mode == "full":

        return cp.arange(
            current_time,
            next_time,
            params.t_step,
        )

    elif params.kl_time_mode == "subsample":

        return cp.arange(
            current_time,
            next_time,
            params.t_step
            * params.kl_time_stride,
        )

    else:

        raise ValueError(
            "Unknown KL time mode."
        )
    
def get_predictions_joint_bias_batch(
    interpolator,
    t_pts,
    candidate_biases,
    bz_samples,
    by_samples,
):
    """
    Predict responses for every

        bias

            ×

        sampled parameter

    simultaneously.

    Parameters
    ----------

    t_pts

        (Nt,)

    candidate_biases

        (Nb,)

    bz_samples

        (Np,)

    by_samples

        (Np,)

    Returns
    -------

    prediction

        (Nb,Np,Nt)
    """

    Nb = len(candidate_biases)
    Np = len(bz_samples)
    Nt = len(t_pts)

    # ------------------------------
    # Broadcast
    # ------------------------------

    T = cp.broadcast_to(
        t_pts[None, None, :],
        (Nb, Np, Nt),
    )

    BIAS = cp.broadcast_to(
        candidate_biases[:, None, None],
        (Nb, Np, Nt),
    )

    BZ = cp.broadcast_to(
        bz_samples[None, :, None],
        (Nb, Np, Nt),
    )

    BY = cp.broadcast_to(
        by_samples[None, :, None],
        (Nb, Np, Nt),
    )

    # Total longitudinal field

    Z = BIAS + BZ

    prediction = interpolator.interpolate(
        T.ravel(),
        Z.ravel(),
        BY.ravel(),
    )

    return prediction.reshape(
        Nb,
        Np,
        Nt,
    )

def get_predictions_joint_grid(
    interpolator,
    t_pts,
    candidate_biases,
    bz_grid,
    by_grid,
):
    """
    Predict the transient response over the ENTIRE joint support
    for EVERY candidate bias.

    Parameters
    ----------
    interpolator : GPUInterpolator

    t_pts : (Nt,)
        Time points used for KL evaluation.

    candidate_biases : (Nb,)
        Candidate control fields.

    bz_grid : (Nz,)
        Longitudinal support.

    by_grid : (Ny,)
        Transverse support.

    Returns
    -------
    prediction : (Nb, Nt, Nz, Ny)

        prediction[b, t, iz, iy]
    """

    Nb = len(candidate_biases)
    Nt = len(t_pts)
    Nz = len(bz_grid)
    Ny = len(by_grid)

    # ---------------------------------------------------------
    # Construct broadcasted query tensors
    #
    # Final shape:
    #
    # (Nb, Nt, Nz, Ny)
    # ---------------------------------------------------------

    T = t_pts[None, :, None, None]

    BIAS = candidate_biases[:, None, None, None]

    BZ = bz_grid[None, None, :, None]

    BY = by_grid[None, None, None, :]

    final_shape = (Nb, Nt, Nz, Ny)

    T = cp.broadcast_to(T, final_shape)

    BIAS = cp.broadcast_to(BIAS, final_shape)

    BZ = cp.broadcast_to(BZ, final_shape)

    BY = cp.broadcast_to(BY, final_shape)

    # Total longitudinal field

    Z = BIAS + BZ

    prediction = interpolator.interpolate(
        T.ravel(),
        Z.ravel(),
        BY.ravel(),
    )

    return prediction.reshape(final_shape)

def calculate_joint_likelihood_mc_gpu(
    observations,
    t_pts,
    candidate_biases,
    bz_grid,
    by_grid,
    interpolator,
    sigma_noise,
    likelihood_mode="Gaussian",
):
    Nb = len(candidate_biases)
    Nz = len(bz_grid)
    Ny = len(by_grid)
    Nt = len(t_pts)

    prediction_grid = get_predictions_joint_grid(
        interpolator,
        t_pts,
        candidate_biases,
        bz_grid,
        by_grid,
    )

    obs = observations[:, :, :, None, None]
    pred = prediction_grid[:, None, :, :, :]
    if likelihood_mode == "Gaussian":
        residual = obs - pred

        SSE = cp.sum(
            residual**2,
            axis=2,
        )

        logL = -SSE/(2*sigma_noise**2)
    elif likelihood_mode == "Cauchy":
        gamma = sigma_noise * 1.177

        log_term = cp.log(
            1
            + residual**2/gamma**2
        )

        logL = -cp.sum(
            log_term,
            axis=2,
        )

    logL = logL - cp.max(logL,axis=(-2,-1),keepdims=True,)
    
    return logL


def expected_information_gain_mc_gpu(
    posterior,
    candidate_biases,
    t_pts,
    interpolator,
    params,
):
    # ---------------------------------
    # Sample posterior
    # ---------------------------------

    bz_samples, by_samples = posterior.sample(
        params.kl_parameter_samples
    )

    # ---------------------------------
    # Predict sampled responses
    # ---------------------------------

    predictions = get_predictions_joint_bias_batch(
        interpolator,
        t_pts,
        candidate_biases,
        bz_samples,
        by_samples,
    )

    # ---------------------------------
    # Generate synthetic observations
    # ---------------------------------

    observations = generate_synthetic_observations_gpu(
        predictions,
        params.sigma_noise_longitudinal,
    )

    # ---------------------------------
    # Likelihood over whole grid
    # ---------------------------------

    logL = calculate_joint_likelihood_mc_gpu(
        observations,
        t_pts,
        candidate_biases,
        posterior.bz_axis,
        posterior.by_axis,
        interpolator,
        params.sigma_noise_longitudinal,
        params.likelihood_mode_longitudinal,
    )

    # ---------------------------------
    # Bayesian update
    # ---------------------------------

    prior_log = posterior.log_weights

    prior_log = prior_log[None, None, :, :]

    posterior_log = prior_log + logL

    posterior_log = normalize_log_weights(
        posterior_log
    )

    # ---------------------------------
    # KL
    # ---------------------------------

    kl = compute_kl_gpu(
        prior_log,
        posterior_log,
    )

    utility = cp.mean(
        kl,
        axis=1,
    )

    return utility

def expected_information_gain_mc_gpu(
    posterior,
    candidate_biases,
    current_time,
    next_time,
    interpolator,
    params,
):
    """
    Monte Carlo approximation of

        E_y [ KL ]

    Parameters
    ----------
    posterior : JointPosterior

    candidate_biases : (Nb,)

    current_time : float

    next_time : float

    interpolator : GPUInterpolator

    params : ParameterContext

    Returns
    -------
    utility : (Nb,)
    """

    # ---------------------------------------------------------
    # Time grid
    # ---------------------------------------------------------

    t_pts = build_prediction_times(
        current_time,
        next_time,
        params,
    )

    # ---------------------------------------------------------
    # Sample hidden parameters
    # ---------------------------------------------------------

    bz_samples, by_samples = posterior.sample(
        params.kl_parameter_samples
    )

    # ---------------------------------------------------------
    # Predict "true" measurements
    # ---------------------------------------------------------

    predictions = get_predictions_joint_bias_batch(
        interpolator,
        t_pts,
        candidate_biases,
        bz_samples,
        by_samples,
    )

    # ---------------------------------------------------------
    # Add measurement noise
    # ---------------------------------------------------------

    observations = generate_synthetic_observations_gpu(
        predictions,
        params.sigma_noise_longitudinal,
    )

    # ---------------------------------------------------------
    # Likelihood over support grid
    # ---------------------------------------------------------

    logL = calculate_joint_likelihood_mc_gpu(
        observations,
        t_pts,
        candidate_biases,
        posterior.bz_axis,
        posterior.by_axis,
        interpolator,
        params.sigma_noise_longitudinal,
        params.likelihood_mode_longitudinal,
    )

    # ---------------------------------------------------------
    # Prior
    # ---------------------------------------------------------

    prior_log = posterior.log_weights

    prior_log = prior_log[None, None, :, :]

    # ---------------------------------------------------------
    # Temporary posterior
    # ---------------------------------------------------------

    posterior_log = prior_log + logL

    posterior_log = normalize_log_weights(
        posterior_log
    )

    # ---------------------------------------------------------
    # KL
    # ---------------------------------------------------------

    kl = compute_kl_gpu(
        prior_log,
        posterior_log,
    )

    # ---------------------------------------------------------
    # Monte Carlo expectation
    # ---------------------------------------------------------

    utility = cp.mean(
        kl,
        axis=1,
    )

    return utility

def choose_next_bias(
    posterior,
    candidate_biases,
    current_time,
    next_time,
    interpolator,
    params,
):
    """
    Choose the candidate bias with maximum
    expected information gain.
    """

    utility = expected_information_gain_mc_gpu(
        posterior,
        candidate_biases,
        current_time,
        next_time,
        interpolator,
        params,
    )

    best = cp.argmax(utility)

    return (
        candidate_biases[best],
        utility,
    )