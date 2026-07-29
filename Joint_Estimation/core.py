from email.mime import base
import os
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
import cupy as cp
# from cupyx.scipy.ndimage import map_coordinates
import time
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional
# from scipy.interpolate import RegularGridInterpolator
# from scipy.ndimage import spline_filter

# Data loading
@dataclass
class DataContext:
    """Bundle of file strings that allow loading of the data we are interested in. 
    
    Stores paths to simulation and experiment data, interpolator cache, and handles
    timestamped result directories.
         
    Attrs:
    - sim_time: Path to simulation time axis csv file
    - sim_freq: Path to simulation frequency axis csv file
    - sim_by: Path to simulation By axis csv file
    - sim_intensities: List of simulation intensity file paths
    - exp_time: Path to experiment time axis csv file
    - exp_freq_axis: Path to experiment frequency bias axis csv file
    - exp_data: Path to experiment probe transmission data csv file
    - interpolator: Path to cached interpolator pickle file (or None)
    - save_path: Base directory for results (timestamped subdirectory auto-created)
    """
    sim_time: str
    sim_freq: str
    sim_by: str
    sim_intensities: list[str]
    exp_time: str
    exp_freq_axis: str
    exp_data: str
    interpolator: Optional[str]
    save_path: str = f"Results/Dataset_10/Test_{datetime.now().date().strftime('%d_%m_%Y')}"
    Sim_Aligned: bool = False
    Exp_Aligned: bool = False

    # Pulse parameters
    sim_pulse_thresh: float = 0.3
    exp_pulse_thresh: float = 0.3


    def __setattr__(self, prop, val):
        if prop == 'save_path' and val is not None:
            if 'save_path' in self.__dict__ and self.__dict__['save_path'] is not None:
                current_path = getattr(self, 'save_path', None)
                val_norm = os.path.normpath(str(val))
                if current_path is not None and os.path.basename(current_path) != val_norm:
                    val = os.path.join(current_path, str(val))
            os.makedirs(val, exist_ok=True)
        super().__setattr__(prop, val)


@dataclass
class ParameterContext:
    """Bundle of runtime parameters for experiments and alt-opt runs.

    This centralizes commonly used knobs with sensible defaults so you can
    pass one object around instead of many separate arguments.

    Longitudinal/Y Estimation (common):
    - curr_time: Start time (s)
    - print_plot: Whether to show intermediate plots
    - zoom_factor: Adaptive zoom factor (> 1)
    - max_time: End time (s)
    - t_step: Measurement step (s)
    - B_unk_bound: Prior bound for unknown field (|B| <= bound)
    - init_resolution: Initial grid resolution
    - sigma_noise: Observation noise stddev
    - f_bias_offset_nuisance: Constant offset added to exp bias axis (Z only)

    AltOpt-only:
    - num_iter: Alternating optimization iterations
    - tol_bz / tol_by: early-stop tolerances
    - patience: consecutive stable iterations required to early-stop
    - Est_First_Z: whether to estimate Z first
    - fixed_bz_estimate / fixed_by_estimate: initial seeds
    """

    # Common experiment parameters
    Test: str = ""
    curr_time: float = 5.0
    # print_plot: bool = False
    # zoom_factor: float = 1.1
    # zoom_trigger_multiple: int = 400
    # zoom_trigger_ratio: float = 0.2
    max_time: float = 70.0
    t_step: float = 0.2
    B_unk_bound_longitudinal: float = 1.0
    B_unk_bound_transverse_lower: float = 0.0
    B_unk_bound_transverse_upper: float = 0.5
    init_resolution_longitudinal: float = 0.01
    init_resolution_transverse: float = 0.005
    sigma_noise_longitudinal: float = 0.4
    sigma_noise_transverse: float = 0.4
    # likelihood_mode_longitudinal: str = "Gaussian"
    # likelihood_mode_transverse: str = "Gaussian"
    f_bias_offset_nuisance: float = 0.0
    # Est_First_Z: bool = False

    # Monte Carlo parameters
    kl_parameter_samples: int = 256      # N_theta
    kl_outcome_samples: int = 64         # N_y

    # Time points used for KL prediction
    kl_time_mode: str = "subsample"           # "full" or "subsample"
    kl_time_points: int = 2
    kl_time_stride: int = 4              # every 4th sample if subsample

    # Pulse parameters
    sim_pulse_thresh: float = 0.3
    exp_pulse_thresh: float = 0.3
    
    # # Adaptive experiment/grid
    # kl_y_grid_size: int = 50

    # # AltOpt parameters
    # num_iter: int = 5
    # tol_bz: float = 1e-8
    # tol_by: float = 1e-8
    # patience: int = 2

    # fixed_bz_estimate: float = 0.0
    # fixed_by_estimate: float = 0.0


    def __post_init__(self):
        self.Test = f"Starting_{self.curr_time}_$\mu$s_T_Step_{self.t_step}_$\mu$s"

    def validate(self) -> tuple[bool, list[str]]:
        """Basic parameter sanity checks.

        Returns (ok, problems).
        """
        problems: list[str] = []
        # if self.zoom_factor <= 1.0:
        #     problems.append("zoom_factor must be > 1.0")
        if self.t_step <= 0:
            problems.append("t_step must be > 0")
        if self.max_time <= self.curr_time:
            problems.append("max_time must be > curr_time")
        if self.B_unk_bound <= 0:
            problems.append("B_unk_bound must be > 0")
        if self.init_resolution_longitudinal <= 0:
            problems.append("init_resolution must be > 0")
        if self.init_resolution_transverse <= 0:
            problems.append("init_resolution must be > 0")
        # if self.sigma_noise < 0:
        #     problems.append("sigma_noise must be >= 0")
        # if self.num_iter <= 0:
        #     problems.append("num_iter must be > 0")
        # if self.patience < 1:
        #     problems.append("patience must be >= 1")
        return (len(problems) == 0, problems)

class GPUInterpolator:
    def __init__(self, t_axis = None, f_axis = None, by_axis = None, data_cube = None, coeffs_cpu = None):
        if data_cube is not None:
            self.t_axis = cp.asarray(t_axis)
            self.f_axis = cp.asarray(f_axis)
            self.by_axis = cp.asarray(by_axis)
            self.data_cube = cp.asarray(data_cube, dtype=cp.float64)
            self.dims = self.data_cube.shape
            self.nt, self.nz, self.ny = self.dims

            self.t_start = t_axis[0]
            self.f_start = f_axis[0]
            self.by_start = by_axis[0]

            self.t_step = t_axis[1] - t_axis[0]
            self.f_step = f_axis[1] - f_axis[0]
            self.by_step = by_axis[1] - by_axis[0]

            # Precompute the maximum valid indices for clamping
            self.max_t_idx = len(t_axis) - 1
            self.max_f_idx = len(f_axis) - 1
            self.max_by_idx = len(by_axis) - 1

            self.kernel = cp.ElementwiseKernel(
                in_params = 'float64 t_idx, float64 z_idx, float64 y_idx, raw float64 cube, int32 nt, int32 nz, int32 ny',
                out_params = 'float64 out',
                operation = """
                    int ti = max(0, min((int)floor(t_idx), nt - 2));
                    int zi = max(0, min((int)floor(z_idx), nz - 2));
                    int yi = max(0, min((int)floor(y_idx), ny - 2));

                    float dt = t_idx - ti;
                    float dz = z_idx - zi;
                    float dy = y_idx - yi;

                    int stride_t = nz*ny;
                    int stride_z = ny;
                    int stride_y = 1;

                    int idx000 = ti*stride_t + zi*stride_z + yi*stride_y;

                    float v000 = cube[idx000];
                    float v001 = cube[idx000 + 1];
                    float v010 = cube[idx000 + stride_z];
                    float v011 = cube[idx000 + stride_z + 1];
                    float v100 = cube[idx000 + stride_t];
                    float v101 = cube[idx000 + stride_t + 1];
                    float v110 = cube[idx000 + stride_t + stride_z];
                    float v111 = cube[idx000 + stride_t + stride_z + 1];

                    float c00 = v000*(1-dy) + v001*dy;
                    float c01 = v010*(1-dy) + v011*dy;
                    float c10 = v100*(1-dy) + v101*dy;
                    float c11 = v110*(1-dy) + v111*dy;

                    float c0 = c00*(1-dz) + c01*dz;
                    float c1 = c10*(1-dz) + c11*dz;

                    out = c0*(1-dt) + c1*dt;
                """,
                name = "trilinear_extrapolation_and_interpolation_kernel"
            )
            
            # Push the finalized cubic spline coefficients to the GPU memory
            self.coeffs_gpu : cp.ndarray

    def to_indices(self, t_query, f_query, by_query):
        t_idx = (t_query - self.t_start) / self.t_step
        f_idx = (f_query - self.f_start) / self.f_step
        f_idx = cp.clip(f_idx, 0, self.nz - 1)
        idx = cp.searchsorted(self.by_axis, by_query, side='right') - 1
        idx = cp.clip(idx, 0, self.ny - 2)
        y0 = self.by_axis[idx]
        y1 = self.by_axis[idx + 1]
        by_idx = idx + (by_query - y0) / (y1 - y0)
        return t_idx, f_idx, by_idx
    
    def interpolate(self, t_query, f_query, by_query):  
        # Note: Query the flattened arrays directly to keep calculations simple and efficient
        t_idx, f_idx, by_idx = self.to_indices(t_query, f_query, by_query)
        coords = cp.stack([t_idx, f_idx, by_idx], axis = 0)
        # predictions = map_coordinates(
        #     self.coeffs_gpu, 
        #     coords, 
        #     order=3, 
        #     mode='nearest', 
        #     prefilter=False 
        # )
        # predictions = map_coordinates(self.data_cube, coords, order=1, mode='nearest')
        return self.kernel(t_idx.astype(cp.float64), f_idx.astype(cp.float64), by_idx.astype(cp.float64), self.data_cube, self.nt, self.nz, self.ny)
        # return cp.clip(predictions, 0.0, 1.0)
    
    def save(self, filepath):
        np.savez(filepath, 
                 t_axis=cp.asnumpy(self.t_axis), 
                 f_axis=cp.asnumpy(self.f_axis), 
                 by_axis=cp.asnumpy(self.by_axis), 
                 metadata=np.array([
                self.t_start, self.t_step,
                self.f_start, self.f_step,
                self.by_start, self.by_step
                ]),
                 data_cube=cp.asnumpy(self.data_cube))
        # base = os.path.splitext(filepath)[0] # "document"
        # new_path = base + ".npy"
        # filepath_1 = new_path
        # if os.path.exists(filepath_1):
        #         print(f"Loading pre-computed cubic spline coefficients from: {filepath_1}")
        #         coeffs_cpu = np.load(filepath_1)
        # else:
        #     print("Cubic spline coefficients not found. Computing IIR filter...")
        #     print("This may take a minute, but will only happen once for this dataset.")
        
        #     # Calculate the coefficients on the CPU using SciPy
        #     coeffs_cpu = spline_filter(self.data_cube.get(), order=3)
        
        #     # Save the numpy array so we bypass this step in future runs
        #     np.save(filepath_1, coeffs_cpu)
        #     print(f"Coefficients saved successfully to: {filepath_1}")
        # self.coeffs_gpu = cp.asarray(coeffs_cpu)
        
    @classmethod
    def load(cls, filepath):
        loaded = np.load(filepath)
        t_axis = cp.asarray(loaded['t_axis'])
        f_axis = cp.asarray(loaded['f_axis'])
        by_axis = cp.asarray(loaded['by_axis'])
        data_cube = cp.asarray(loaded['data_cube'], dtype = cp.float64)
        meta = loaded['metadata']

        base = os.path.splitext(filepath)[0] # "document"
        new_path = base + ".npy"
        filepath_1 = new_path
        #i dont need to pass time, by, bz because all that matters are the min and steps, which have been passed
        instance = cls()
        # coeffs_cpu = np.load(filepath_1)
        instance.by_axis = by_axis
        instance.data_cube = data_cube
        # instance.coeffs_gpu = cp.asarray(coeffs_cpu)
        instance.nt, instance.nz, instance.ny = data_cube.shape
        instance.t_start, instance.t_step = float(meta[0]), float(meta[1])
        instance.f_start, instance.f_step = float(meta[2]), float(meta[3])
        instance.by_start, instance.by_step = float(meta[4]), float(meta[5])
        instance.max_t_idx, instance.max_f_idx, instance.max_by_idx = len(t_axis) - 1, len(f_axis) - 1, len(by_axis) - 1    
        instance.kernel = cp.ElementwiseKernel(
            in_params = 'float64 t_idx, float64 z_idx, float64 y_idx, raw float64 cube, int32 nt, int32 nz, int32 ny',
            out_params = 'float64 out',
            operation = """
                int ti = max(0, min((int)floor(t_idx), nt - 2));
                int zi = max(0, min((int)floor(z_idx), nz - 2));
                int yi = max(0, min((int)floor(y_idx), ny - 2));
                float dt = t_idx - ti;
                float dz = z_idx - zi;
                float dy = y_idx - yi;
                int stride_t = nz*ny;
                int stride_z = ny;
                int stride_y = 1;
                int idx000 = ti*stride_t + zi*stride_z + yi*stride_y;
                float v000 = cube[idx000];
                float v001 = cube[idx000 + 1];
                float v010 = cube[idx000 + stride_z];
                float v011 = cube[idx000 + stride_z + 1];
                float v100 = cube[idx000 + stride_t];
                float v101 = cube[idx000 + stride_t + 1];
                float v110 = cube[idx000 + stride_z + stride_t];
                float v111 = cube[idx000 + stride_z + stride_t + 1];
                float c00 = v000*(1-dy) + v001*dy;
                float c01 = v010*(1-dy) + v011*dy;
                float c10 = v100*(1-dy) + v101*dy;
                float c11 = v110*(1-dy) + v111*dy;
                float c0 = c00*(1-dz) + c01*dz;
                float c1 = c10*(1-dz) + c11*dz;
                out = c0*(1-dt) + c1*dt;
                """,
                name = "trilinear_extrapolation_and_interpolation_kernel"
            )
        return t_axis, f_axis, by_axis, instance

# default object: 
DEFAULT_PARAMS = ParameterContext()

DATA_DIR = r"DataFiles_to_Dinesh_Pranav\Data_files\Simulation\Dataset_10"
DATASET3 = DataContext(
    sim_time = os.path.join(DATA_DIR, "t_array.csv"),
    sim_freq = os.path.join(DATA_DIR, "delz_MHz.csv"),
    sim_by = os.path.join(DATA_DIR, "dely_MHz.csv"),
    sim_intensities = [],
    exp_time = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\t_exp.csv",
    exp_freq_axis = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\Y_MHz_Exp.csv",
    exp_data = r"DataFiles_to_Dinesh_Pranav\Data_files\Experiment\Probe_trans_Exp.csv",
    interpolator=r"DataFiles_to_Dinesh_Pranav\Data_files\Simulation\Dataset_10\gpu_sim_interpolator_new_normalisation_linear_0.3_threshold.npz",
    Sim_Aligned = False,
    Exp_Aligned = False
)



## Preprocessing
def find_pulse_start(trace, t_axis, threshold):
    idx = np.where(trace > threshold)[0]
    if len(idx) > 0:
        return t_axis[idx[0]], idx[0]
    return t_axis[0], 0


def load_simulation_cube(config: DataContext, Aligned=DATASET3.Sim_Aligned):
    """Load simulation axes and cube using DataContext paths."""
    print("...Loading Simulation Cube Data...")
    to = time.time()
    t_axis = pd.read_csv(config.sim_time, header=None).values.flatten()
    f_axis = pd.read_csv(config.sim_freq, header=None).values.flatten()
    by_axis = pd.read_csv(config.sim_by, header=None).values.flatten()

    print(
        f"  > Axes Loaded: Time[{len(t_axis)}], Freq[{len(f_axis)}], By[{len(by_axis)}]"
    )

    t_sim_0, _ = 0.0, 0
    t_axis_0 = t_axis
    Aligned = config.Sim_Aligned
    matrix_list = []
    for fname in config.sim_intensities:
        data = pd.read_csv(fname, header=None).values

        if data.shape != (len(t_axis), len(f_axis)):
            data = data.T
        if not Aligned:
            if len(matrix_list) == 0:
                scale = 1 / (np.max(data) - np.min(data))
                calibrated_data = (data - np.min(data)) * scale
                calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
                t_sim_0, _ = find_pulse_start(calibrated_data[:, 0], t_axis, config.sim_pulse_thresh)

            t_axis_0 = t_axis[_:] - t_sim_0

        data = data[_:, :]
        scale = 1 / (np.max(data) - np.min(data))
        calibrated_data = (data - np.min(data)) * scale
        calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
        matrix_list.append(calibrated_data)

    cube = np.stack(matrix_list, axis=2)
    print(f"  > Sim Start: {t_sim_0:.4f}s")
    print(f"  > Cube Built. Final Shape: {cube.shape}")
    print("Time taken to load sim data:", time.time() - to)
    return t_axis_0, f_axis, by_axis, cube


def load_experiment(config: DataContext, Aligned=DATASET3.Exp_Aligned):
    to = time.time()
    print(f"Loading Experiment from {config.exp_data}...")
    t = pd.read_csv(config.exp_time, header=None).values.flatten()
    f_bias = pd.read_csv(config.exp_freq_axis, header=None).values.flatten()
    raw_data = pd.read_csv(config.exp_data, header=None).values

    if raw_data.shape != (len(t), len(f_bias)):
        raw_data = raw_data.T

    if not Aligned:
        scale = 1 / (np.max(raw_data) - np.min(raw_data))
        calibrated_data = (raw_data - np.min(raw_data)) * scale
        calibrated_data = np.clip(calibrated_data, 0.0, 1.0)
        t_exp_0, __ = find_pulse_start(calibrated_data[:, 0], t, config.exp_pulse_thresh)

        t = t[__:] - t_exp_0
        raw_data = raw_data[__:,]
        print(f"  > Exp Start: {t_exp_0:.4f}s")
    
    #this is for the dataset from 13/02 only FIXME
    # t = t * 1e6
    t_index = np.where(np.asarray(t)>70)[0][0]
    f_index_1 = np.where(np.asarray(f_bias)<-1.5)[0][0]
    f_index_2 = np.where(np.asarray(f_bias)>1.5)[0][0]
    t = t[:t_index]
    raw_data = raw_data[:t_index,:]
    calibrated_data= calibration(raw_data, np.min(raw_data), np.max(raw_data))
   
    print(time.time() - to, "was the time taken to load exp data.")
    return t, f_bias, calibrated_data


def calibration(data, v_dark, v_max):
    scale = 1 / (v_max - v_dark)
    calibrated_data = (data - v_dark) * scale
    return calibrated_data


#Plotting
base_cmap = plt.get_cmap('inferno')

# Sample the colormap from 0.2 to 1.0, explicitly skipping the darkest 20%
color_array = base_cmap(np.linspace(0.2, 1.0, 256))

# Create the new, brighter colormap
custom_inferno = mcolors.LinearSegmentedColormap.from_list('inferno_bright', color_array)

#TODO Make sure that this code is refactored for plotting KL divergence GIFS our joint estimator implimentation
def plot_heat_and_surface(
    X_vec,
    Y_vec,
    Z_arr,
    title_prefix="Data",
    trajectory_mode=False,
    traj_bias=[],
    curr_time=5.0,
    save_path = '',
    format = "svg"
):
    Y_vec = Y_vec #FIXME Mhz to Gauss conversion for plotting
    Xg, Yg = np.meshgrid(X_vec, Y_vec, indexing="ij")  # note: meshgrid order (cols = y)
    # heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    # 2. Adjust the Gradient Scaling
    # Use PowerNorm (gamma < 1) to stretch the color gradient over the lower/middle values
    # or use LogNorm() if your loss landscape varies by multiple orders of magnitude.

    norm = mcolors.PowerNorm(gamma=1, vmin=Z_arr.min(), vmax=Z_arr.max())
    im = ax.pcolormesh(Xg, Yg, Z_arr, shading="nearest", cmap = custom_inferno, norm = norm)
    ax.set_xlabel("$t$ ($\mathrm{\mu s}$)")
    ax.set_ylabel("$B_c$ ($\mathrm{\mu T}$)")
    if trajectory_mode:
        curr_time_index = np.searchsorted(X_vec, curr_time)
        traj_bias = np.repeat(
            traj_bias, (X_vec.shape[0] - curr_time_index - 1) // len(traj_bias)
        )
        traj_bias = np.asarray(traj_bias)/0.7 #FIXME Mhz to Gauss conversion for plotting
        ax.plot(
            X_vec[curr_time_index] + X_vec[: len(traj_bias)],
            traj_bias,
            color="red",
            marker="o",
            markersize=5,
            label="Trajectory",
        )
    ax.tick_params(axis='both', which='both', top=True, right=True,direction='in', length=6)#, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor('black')
    cbar = plt.colorbar(im, label="$\mid I_{\mathrm{exp}}(t, B_c) - I_{\mathrm{sim}}(t, B_c; \hat{B}_y, \hat{B}_z)\mid$")
    cbar.outline.set_linewidth(1.2)
    cbar.outline.set_edgecolor('black')
    cbar.ax.tick_params(direction='in', length=6)#, width=1.5)
    # plt.title(f"{title_prefix}", fontsize = 10)
    plt.tight_layout()
    if save_path != "": 
        plt.savefig(save_path, format = format, dpi = 1200); plt.show();plt.close()
    if save_path == "": plt.show()
    # 3D surface (might be heavy)
    # fig = plt.figure(figsize=(9, 6))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(Yg, Xg, Z_arr, cmap="viridis", linewidth=0, antialiased=True)
    # ax.set_xlabel("$B_c$ ($\mu T$)")
    # ax.set_ylabel("Time ($\mu$s)")
    # ax.set_zlabel("Probe Transmitted Intensity")
    # plt.title(f"{title_prefix}", fontsize = 10)
    # plt.tight_layout()
    # if save_path == "": plt.show()
    # if save_path != "": plt.show();plt.close()

def animation_posterior(posteriors, bz_grid, by_grid, save_path):
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cbar = None

    # Animation update function
    def update(frame):
        nonlocal cbar
        ax.clear()
        # Assuming bgrid and posteriors are lists/arrays of length 90
        x = by_grid[frame]*100/0.7 #Conversion from Mhz to Gauss
        y = bz_grid[frame]*100/0.7
        posterior = posteriors[frame]

        X, Y = np.meshgrid(x, y, indexing = "ij")
        
        # Update data
        norm = mcolors.PowerNorm(gamma=1, vmin=posterior.min(), vmax=posterior.max())
        heat = ax.pcolormesh(X, Y, posterior, cmap=custom_inferno, shading='auto', norm = norm)
        
        # Rescale axes dynamically to fit the new zoomed/peaked distribution
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel(r"$B_y$ $\mu$T"); ax.set_ylabel(r"$B_z$ $\mu$T")
        ax.set_title(f"Posterior Step {frame}")
        if cbar is None:
            cbar = fig.colorbar(heat, ax=ax, label="PDF Value")
            cbar.outline.set_linewidth(1.2)
            cbar.outline.set_edgecolor('black')
            cbar.ax.tick_params(direction='in', length=6)
        else:
            # Updates existing colorbar scale matching the new distribution 
            cbar.update_normal(heat)
        return [heat]

    # Create animation (interval in milliseconds)
    anim = FuncAnimation(fig, update, frames=len(posteriors), blit=False, interval=100)

    # Option 2: Save as GIF (No external dependencies usually)
    anim.save(save_path, writer='pillow', fps=4)
    plt.close(fig)

#TODO Make sure that this code is refactored for plotting KL divergence GIFS our joint estimator implimentation
def animation_kl(expectedkl, f_bias_axis, save_path):
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 6))
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel("Bias Field")
    ax.set_ylabel("Expected KL Divergence")
    ax.grid(True)

    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation update function
    def update(frame):
        # Assuming bgrid and posteriors are lists/arrays of length 90
        x = f_bias_axis
        y = expectedkl[frame]
        
        # Update data
        line.set_data(x, y)
        
        # Rescale axes dynamically to fit the new zoomed/peaked distribution
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, y.max() * 1.1)
        ax.set_xlabel("Longitudinal Bias Field ($\mu T$)")
        ax.set_title(f"Expected_KL {frame}")
        return line,

    # Create animation (interval in milliseconds)
    anim = FuncAnimation(fig, update, frames=len(expectedkl), 
                        init_func=init, blit=False, interval=100)

    # Option 2: Save as GIF (No external dependencies usually)
    anim.save(save_path, writer='pillow', fps=4)
    anim.close()


    # interpolator and interpolation validator for Y field estimation


def get_final_interpolator(config: Optional[DataContext], params: ParameterContext):
    # if the config DataContext stringblob has an interpolator path, use that to load the interpolator otherwise generate.
    print("...Building Final Interpolator...")
    if not os.path.exists(config.interpolator):
        to = time.time()
        t_sim, f_sim, by_sim, cube = load_simulation_cube(config)
        full_interp = GPUInterpolator(t_sim, f_sim, by_sim, cube)
        full_interp.save(config.interpolator)
        t_sim, f_sim, by_sim = cp.asarray(t_sim), cp.asarray(f_sim), cp.asarray(by_sim)
    else:
        to = time.time()
        t_sim, f_sim, by_sim, full_interp = GPUInterpolator.load(config.interpolator)  
    print("Time taken to load Interpolator:", time.time() - to)
    return t_sim, f_sim, by_sim, full_interp
