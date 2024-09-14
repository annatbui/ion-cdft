import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv


# Enable or disable Tensor Float 32 Execution
tf.config.experimental.enable_tensor_float_32_execution(False)


def generate_windows(array, bins):
    """
    Generate sliding windows for the input array with a given bin size.

    Parameters:
    - array (np.ndarray): Input array.
    - bins (int): Number of bins on each side of the central bin.
    - mode (str): Padding mode for np.pad (default is "wrap").

    Returns:
    - np.ndarray: Array of sliding windows.
    """
    padded_array = np.pad(array, bins, mode="wrap")
    windows = np.empty((len(array), 2 * bins + 1))
    for i in range(len(array)):
        windows[i] = padded_array[i:i + 2 * bins + 1]
    return windows

def c1_onetype(model, density_profile, input_bins, dx=0.01, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    window_bins = (input_bins - 1) // 2
    rho_windows = generate_windows(density_profile, window_bins).reshape(density_profile.shape[0], input_bins, 1)
    
    if return_c2:
        rho_windows = tf.Variable(rho_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rho_windows)
            result = model(rho_windows)
        jacobi_windows = tape.batch_jacobian(result, rho_windows).numpy().squeeze() / dx
        c1_result = result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1_result, jacobi_windows
        
        c2_result = np.row_stack([
            np.roll(np.pad(jacobi_windows[i], (0, density_profile.shape[0] - input_bins)), i - window_bins) 
            for i in range(density_profile.shape[0])
        ])
        return c1_result, c2_result
    
    if output_dict:
        return model.predict_on_batch(rho_windows)["c1"].flatten()
    
    return model.predict_on_batch(rho_windows).flatten()

def c1_twotype(model_H, model_O, rho_H, rho_O, input_bins, dx=0.03, return_c2=False, output_dict=False):
    """
    Infer the one-body direct correlation profile from a given density profile 
    using a neural correlation functional.

    Parameters:
    - model (tf.keras.Model): The neural correlation functional.
    - density_profile (np.ndarray): The density profile.
    - dx (float): The discretization of the input layer of the model.
    - input_bins (int): Number of input bins for the model.
    - return_c2 (bool or str): If False, only return c1(x). If True, return both 
                               c1 as well as the corresponding two-body direct 
                               correlation function c2(x, x') which is obtained 
                               via autodifferentiation. If 'unstacked', give c2 
                               as a function of x and x-x', i.e., as obtained 
                               naturally from the model.

    Returns:
    - np.ndarray: c1(x) or (c1(x), c2(x, x')) depending on the value of return_c2.
    """
    window_bins = (input_bins - 1) // 2
    rhoH_windows = generate_windows(rho_H, window_bins).reshape(rho_H.shape[0], input_bins, 1)
    rhoO_windows = generate_windows(rho_O, window_bins).reshape(rho_O.shape[0], input_bins, 1)
    
    if return_c2:
        rhoH_windows = tf.Variable(rhoH_windows)
        rhoO_windows = tf.Variable(rhoO_windows)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(rhoO_windows)
            tape.watch(rhoH_windows)
            H_result = model_H(rhoH_windows, rhoO_windows)
            O_result = model_O(rhoH_windows, rhoO_windows)
        jacobi_windows_HH = tape.batch_jacobian(H_result, rhoH_windows).numpy().squeeze() / dx
        jacobi_windows_HO = tape.batch_jacobian(H_result, rhoO_windows).numpy().squeeze() / dx
        jacobi_windows_OO = tape.batch_jacobian(O_result, rhoO_windows).numpy().squeeze() / dx
        jacobi_windows_OH = tape.batch_jacobian(O_result, rhoH_windows).numpy().squeeze() / dx
       
        c1H_result = H_result.numpy().flatten()
        c1O_result = O_result.numpy().flatten()
        
        if return_c2 == "unstacked":
            return c1H_result, c1O_result, jacobi_windows_HH, jacobi_windows_HO, jacobi_windows_OO, jacobi_windows_OH
        
        c2_result_HH = np.row_stack([
            np.roll(np.pad(jacobi_windows_HH[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_HO = np.row_stack([
            np.roll(np.pad(jacobi_windows_HO[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_OO = np.row_stack([
            np.roll(np.pad(jacobi_windows_OO[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        c2_result_OH = np.row_stack([
            np.roll(np.pad(jacobi_windows_OH[i], (0, rho_H.shape[0] - input_bins)), i - window_bins) 
            for i in range(rho_H.shape[0])
        ])
        
        return (c1H_result, c2_result_HH, c2_result_HO), (c1O_result, c2_result_OO, c2_result_OH)
    
    
    if output_dict:
        c1H_result = model_H.predict_on_batch([rhoH_windows, rhoO_windows])["c1_H"].flatten()
        c1O_result = model_O.predict_on_batch([rhoO_windows, rhoH_windows])["c1_O"].flatten()
    else:
        c1H_result = model_H.predict_on_batch([rhoH_windows, rhoO_windows]).flatten()
        c1O_result = model_O.predict_on_batch([rhoO_windows, rhoH_windows]).flatten()
    return c1H_result, c1O_result

def write_profile(filename, centers, densities):
    """
    Write the density profile to a file.

    Parameters:
    - filename (str): Output file name.
    - centers (np.ndarray): Bin centers.
    - densities (np.ndarray): Density values.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(["xbins", "rho"])
        for center, density in zip(centers, densities):
            writer.writerow([f"{center:.4f}", f"{density:.20f}"])


def pad_pbc(xbins, muloc):
    """
    Pad arrays z and muloc_z for periodic boundary conditions (PBC).
    
    Returns:
    --------
    tuple
        Tuple containing z (padded array of positions), muloc_z (padded local chempot), and L (length scale).
    """
    muloc_z = muloc
    z = xbins - xbins[0] # shift
    dz = z[1] - z[0]
    z = np.append(z, z[-1] + dz)
    muloc_z = np.append(muloc_z, 0.5*(muloc_z[-1] + muloc_z[0]))
    L = z[-1] - z[0]
    
    return z, muloc_z, L
