import numba as nb
import numpy as np
import scipy.constants as const


def compute_wave_numbers(N, dz):
    """
    Compute the wave numbers for a Fourier transform.
    """
    if N % 2 == 0:
        k = np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0)))
    else:
        k = np.concatenate((np.arange(0, (N-1)//2 + 1), np.arange(-(N-1)//2, 0)))
    k = k / (N * dz) * 2 * np.pi
    return k

@nb.njit(fastmath=True, parallel=True)
def fourier_transform(z, f_z, k):
    """
    Compute Fourier transform of f_z using Numba for speedup.

    Parameters:
    k (numpy.ndarray): Array of wave numbers.
    f_z (numpy.ndarray): Array to FT.
    z (numpy.ndarray): Array of spatial positions.

    Returns:
    numpy.ndarray: Fourier transform evaluated at points k.
    """
    N = len(z)
    
    f_k = np.zeros(N, dtype=np.complex128)

    for i in nb.prange(N):
        sum_val = 0.0 + 0.0j
        for j in range(N):
            sum_val += f_z[j] * np.exp(-1j * k[i] * z[j])
        f_k[i] = sum_val

    return k, f_k


@nb.njit(fastmath=True, parallel=True)
def inverse_fourier_transform(k, f_k, z):
    """
    Compute inverse Fourier transform of f_k using Numba for speedup.

    Parameters:
    k (numpy.ndarray): Array of wave numbers.
    f_k (numpy.ndarray): Array of Fourier coefficients.
    z (numpy.ndarray): Array of spatial positions.

    Returns:
    numpy.ndarray: Inverse Fourier transform evaluated at points z.
    """
    N = len(k)
    f_z = np.zeros(N, dtype=np.complex128)
    
    for i in nb.prange(N):
        sum_val = 0.0 + 0.0j
        for j in range(N):
            sum_val += f_k[j] * np.exp(1j * k[j] * z[i])
        f_z[i] = sum_val / N

    return f_z

        
@nb.njit(fastmath=True, parallel=True)
def restructure_electrostatic_potential(n_k, k, z, kappa_inv):
    """
    Compute the convolution of n_k with the Coulomb kernel gaussian.

    Parameters:
    n_k (numpy.ndarray): The Fourier-transformed charge density array.
    k (numpy.ndarray): The wavevector array.
    z (numpy.ndarray): The spatial coordinate array.
    kappa_inv (float): The standard deviation of the Gaussian kernel.

    Returns:
    numpy.ndarray: The convolved result in the spatial domain.
    """
    
    # Identify non-zero k values
    nonzero_indices =  np.where(k != 0)[0]
    k_nonzero = k[nonzero_indices]
    n_k_nonzero = n_k[nonzero_indices]

    # Compute the Gaussian and Coulomb terms
    gaussian_term = np.exp(-k_nonzero**2 * kappa_inv**2 / 4)
    coulomb_term = 4 * np.pi / k_nonzero**2 

    # Initialize the result array
    phi_z = np.zeros_like(z, dtype=np.complex128)

    # Compute the convolution using vectorized operations
    for i in nb.prange(len(z)):
        exponent_term = np.exp(1j * k_nonzero * z[i])
        phi_z[i] = np.sum(coulomb_term * n_k_nonzero * exponent_term * gaussian_term)

    return (phi_z/len(z)).real


@nb.njit(fastmath=True, parallel=True)
def restructure_electric_field(n_k, k, z, kappa_inv):
    """
    Compute the convolution of n_k with the Coulomb kernel gaussian for electric field.

    Parameters:
    n_k (numpy.ndarray): The Fourier-transformed charge density array.
    k (numpy.ndarray): The wavevector array.
    z (numpy.ndarray): The spatial coordinate array.
    kappa_inv (float): The standard deviation of the Gaussian kernel.

    Returns:
    numpy.ndarray: The convolved result in the spatial domain for the electric field.
    """
    
    # Identify non-zero k values
    nonzero_indices =  np.where(k != 0)[0]
    k_nonzero = k[nonzero_indices]
    n_k_nonzero = n_k[nonzero_indices]

    # Compute the Gaussian and Coulomb terms for the electric field
    gaussian_term = np.exp(-k_nonzero**2 * kappa_inv**2 / 4)
    coulomb_term = 4 * np.pi / k_nonzero

    # Initialize the result array
    e_field_z = np.zeros_like(z, dtype=np.complex128)

    # Compute the convolution using vectorized operations
    for i in nb.prange(len(z)):
        exponent_term = np.exp(1j * k_nonzero * z[i])
        e_field_z[i] = np.sum(coulomb_term * n_k_nonzero * exponent_term * gaussian_term)

    return (-1j * e_field_z / len(z)).real


def calculate_prefactor(temp, dielectric):
    """
    Calculate the prefactor for electrostatic potential calculations.

    Parameters:
    - temp (float): Temperature in Kelvin (K).
    - dielectric (float): The dielectric constant of the medium (dimensionless).

    Returns:
    - float: The calculated prefactor.
    """
    
    ang2m = 1e-10
    beta = 1 / (const.Boltzmann * temp)
    return beta * (const.elementary_charge)**2 / (4 * const.pi * const.epsilon_0 * dielectric * ang2m)


# some functions from FFT packages for cross-checking

def fast_compute_wave_numbers(z, dz):
    N = len(z)
    k = np.fft.fftfreq(N, d=dz) * 2 * np.pi
    return k

def fast_inverse_fourier_transform(k, f_k, z):
    """
    Compute inverse Fourier transform of f_k using numpy's ifft.

    """
    f_z = np.fft.ifft(f_k) 
    return f_z


def fast_fourier_transform(z, rho_z):
    """
    Compute Fourier transform of rho_z use numpy's fft   
    """

    N = len(z)
    dz = z[1] - z[0]
    k = np.fft.fftfreq(N, d=dz) * 2 * np.pi
    rho_k = np.fft.fft(rho_z)
    
    return k, rho_k
