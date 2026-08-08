"""
Tensor conventions
------------------

Prediction : (Nt, Nz, Nbias, Ny) 

Residual : (Nt, Nz, Nbias, Ny) 
 
Likelihood : (Nz, Ny) 

Posterior : (Nz, Ny) 
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
    bz_grid : (Nz,) 
    bias : (Nbias,) 
    by_grid : (Ny,)

    Returns
    -------
    prediction : (Nt, Nz, Nbias,Ny) 
    """

    Nt = len(t_pts)
    Nz = len(bz_grid)
    # Nbias, how are we going to query this grid? bias set points, plus minus least count, or do we consider more values in between?
    Nbias = len(bias)
    Ny = len(by_grid)

    # ---------------------------------------------------------
    # Broadcast everything
    # ---------------------------------------------------------

    T = t_pts[:, None, None, None]

    Z = bz_grid[None, :, None, None] + bias[None, None, :, None]

    Y = by_grid[None, None, None, :]

    final_shape = (Nt, Nz, Nbias, Ny)

    T = cp.broadcast_to(T, final_shape)

    Z = cp.broadcast_to(Z, final_shape)

    Y = cp.broadcast_to(Y, final_shape)

    prediction = interpolator.interpolate(
        T.ravel(),
        Z.ravel(),
        Y.ravel(),
    )
    del T,Z,Y
    return prediction.reshape(final_shape)


def calculate_joint_likelihood_gpu(
    measurement,
    t_pts,
    bias, 
    LC_bias,
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
    logL : (Nz, Ny) 
    """
    
    bias_array = cp.linspace(bias - LC_bias, bias + LC_bias, 3) #right now set to 3 points, we can make this more complicated later FIXME 
    prediction = get_predictions_joint(
        interpolator,
        t_pts,
        bz_grid,
        by_grid,
        bias_array, #need to include for 3 point array
    )
    del bias_array

    if likelihood_mode == "Gaussian":

        residual = cp.subtract(
                    measurement[:, None, None, None],
                    prediction
        )

        del prediction

        SSE = cp.sum(
            residual * residual,
            axis=0,
        )

        del residual

        logL = -SSE / (2 * sigma_noise**2)

        del SSE
        print("shape of the log likelihood",logL.shape)
        cp.exp(logL, out = logL)
        logL = cp.sum(logL, axis = 1)
        cp.log(logL, out = logL)

    elif likelihood_mode == "Cauchy":

        gamma = sigma_noise * 1.177

        residual = cp.subtract(
                            measurement[:, None, None, None],
                            prediction
        )

        del prediction 

        log_term = cp.log(
            1
            + residual * residual / gamma**2,
        )

        del residual

        logL = cp.sum(
            -1 * log_term,
            axis=0)
        del log_term
        cp.exp(logL, out  = logL)
        logL = cp.sum(logL, axis = 1)
        cp.log(logL, out = logL)
        


    else:

        raise ValueError(
            f"Unknown likelihood mode {likelihood_mode}"
        )

    # numerical stability

    cp.subtract(logL, cp.max(logL), out = logL)
    return logL