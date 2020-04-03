import os
from typing import Any, List, Tuple, Dict, Union, Optional, cast
from warnings import warn
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.fft import rfftfreq
from scipy.interpolate import interp1d
from chemicalc import exception as e


data_dir: Path = Path(os.path.dirname(__file__)).joinpath("data")
data_dir.mkdir(exist_ok=True)
alpha_el: List[str] = ["O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti"]


def find_nearest(array: Union[List[float], np.ndarray], value: float) -> float:
    """
    ToDo: DocString
    :param Union[List[float], np.ndarray] array: list or array of floats to search
    :param float value: value that you wish to find
    :return float: entry in array that is nearest to value
    """
    if not isinstance(array, (np.ndarray, list)):
        raise TypeError("array must be a np.ndarray or list")
    if isinstance(array, list):
        array = np.asarray(array)
        array = cast(np.ndarray, array)
    idx = (np.abs(array - value)).argmin()
    return float(array[idx])


def find_nearest_idx(array: Union[List[float], np.ndarray], value: float) -> int:
    """
    ToDo: DocString
    :param Union[List[float], np.ndarray] array: list or array of floats to search
    :param float value: value that you wish to find
    :return int: index of entry in array that is nearest to value
    """
    if not isinstance(array, (np.ndarray, list)):
        raise TypeError("array must be a np.ndarray or list")
    if isinstance(array, list):
        array = np.asarray(array)
        array = cast(np.ndarray, array)
    idx = int((np.abs(array - value)).argmin())
    return idx


def generate_wavelength_template(
    start_wavelength: float,
    end_wavelength: float,
    resolution: float,
    res_sampling: float,
    truncate: bool = False,
) -> np.ndarray:
    """
    Generate wavelength array with fixed resolution and wavelength sampling.
    :param float start_wavelength: minimum wavelength of spectra to include
    :param float end_wavelength: maximum wavelength of spectra to include
    :param float resolution: resolving power of instrument (R = lambda / delta lambda)
    :param float res_sampling: pixels per resolution element
    :param bool truncate: If true, drop final pixel for which lambda > end_wavelength
    :return np.ndarray: wavelength grid of given resolution between start and end wavelengths
    """
    if not all(
        isinstance(i, (int, float))
        for i in [start_wavelength, end_wavelength, resolution, res_sampling]
    ):
        raise TypeError("Input quantities must be int or float")
    if not all(
        i > 0 for i in [start_wavelength, end_wavelength, resolution, res_sampling]
    ):
        raise ValueError("Input quantities must be > 0")
    if start_wavelength > end_wavelength:
        raise ValueError("start_wavelength greater than end_wavelength")
    wavelength_tmp = [start_wavelength]
    wavelength_now = start_wavelength

    while wavelength_now < end_wavelength:
        wavelength_now += wavelength_now / (resolution * res_sampling)
        wavelength_tmp.append(wavelength_now)
    wavelength_template = np.array(wavelength_tmp)

    if truncate:
        wavelength_template = wavelength_template[:-1]
    return wavelength_template


def convolve_spec(
    wave: np.ndarray,
    spec: np.ndarray,
    resolution: float,
    outwave: np.ndarray,
    res_in: Optional[float] = None,
) -> np.ndarray:
    """
    Convolves spectrum to lower resolution and samples onto a new wavelength grid
    :param np.ndarray wave: input wavelength array
    :param np.ndarray spec: input spectra array (may be 1 or 2 dimensional)
    :param float resolution: Resolving power to convolve down to (R = lambda / delta lambda)
    :param np.ndarray outwave: wavelength grid to sample onto
    :param Optional[float] res_in: Resolving power of input spectra
    :return np.ndarray: convolved spectra array
    """
    if not all(isinstance(i, np.ndarray) for i in [wave, spec, outwave]):
        raise TypeError("wave, spec, and outwave must be np.ndarray")
    if not isinstance(resolution, (int, float)):
        raise TypeError("resolution must be an int or float")
    if spec.ndim == 1:
        if spec.shape[0] != wave.shape[0]:
            raise ValueError("spec and wave must be the same length")
    elif spec.ndim == 2:
        if spec.shape[1] != wave.shape[0]:
            raise ValueError("spec and wave must be the same length")
    if not (wave.min() < outwave.min() and wave.max() > outwave.max()):
        raise ValueError("outwave extends beyond input wave")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be sorted")
    if not np.all(np.diff(outwave) > 0):
        raise ValueError("outwave must be sorted")

    sigma_to_fwhm = 2.355
    width = resolution * sigma_to_fwhm
    sigma_out = (resolution * sigma_to_fwhm) ** -1
    if res_in is None:
        sigma_in = 0.0
    else:
        if not isinstance(res_in, (int, float)):
            raise TypeError("res_in must be an int or float")
        if res_in < resolution:
            raise ValueError("Cannot convolve to a higher resolution")
        sigma_in = (res_in * sigma_to_fwhm) ** -1

    # Trim Wavelength Range
    nsigma_pad = 20.0
    wlim = np.array([outwave.min(), outwave.max()])
    wlim *= 1 + nsigma_pad / width * np.array([-1, 1])
    mask = (wave > wlim[0]) & (wave < wlim[1])
    wave = wave[mask]
    if spec.ndim == 1:
        spec = spec[mask]
    elif spec.ndim == 2:
        spec = spec[:, mask]
    else:
        raise ValueError("spec cannot have more than 2 dimensions")

    # Make Convolution Grid
    wmin, wmax = wave.min(), wave.max()
    nwave = wave.shape[0]
    nwave_new = int(2 ** (np.ceil(np.log2(nwave))))
    lnwave_new = np.linspace(np.log(wmin), np.log(wmax), nwave_new)
    wave_new = np.exp(lnwave_new)
    fspec = interp1d(wave, spec, bounds_error=False, fill_value="extrapolate")
    spec = fspec(wave_new)
    wave = wave_new

    # Convolve via FFT
    sigma = np.sqrt(sigma_out ** 2 - sigma_in ** 2)
    invres_grid = np.diff(np.log(wave))
    dx = np.median(invres_grid)
    ss = rfftfreq(nwave_new, d=dx)
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    spec_ff = np.fft.rfft(spec)
    ff_tapered = spec_ff * taper
    spec_conv = np.fft.irfft(ff_tapered)

    # Interpolate onto outwave
    fspec = interp1d(wave, spec_conv, bounds_error=False, fill_value="extrapolate")
    return fspec(outwave)


def doppler_shift(
    wave: np.ndarray, spec: np.ndarray, rv: float, bounds_warning: bool = True
) -> np.ndarray:
    """
    ToDo: Unit Tests
    ToDo: Fix Boundary Issues
    Apply doppler shift to spectra and resample onto original wavelength grid
    :param np.ndarray wave: input wavelength array
    :param np.ndarray spec: input spectra array
    :param float rv: Radial Velocity (km/s)
    :param bool bounds_warning: warn about boundary issues?
    :return np.ndarray: Doppler shifted spectra array
    """
    if not all(isinstance(i, np.ndarray) for i in [wave, spec]):
        raise TypeError("wave and spec must be np.ndarray")
    if not isinstance(rv, (int, float)):
        raise TypeError("rv must be an int or float")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("wave must be sorted")
    if rv < 0:
        raise ValueError("rv must be > 0")
    c = 2.99792458e5  # km/s
    doppler_factor = np.sqrt((1 - rv / c) / (1 + rv / c))
    new_wavelength = wave * doppler_factor
    shifted_spec = np.interp(new_wavelength, wave, spec)
    shifted_spec[(wave < new_wavelength.min()) | (wave > new_wavelength.max())] = np.nan
    if bounds_warning:
        if rv > 0:
            warn(
                f"Spectra for wavelengths below {new_wavelength.min()} are undefined; set to np.nan",
                UserWarning,
            )
        if rv < 0:
            warn(
                f"Spectra for wavelengths above {new_wavelength.max()} are undefined; set to np.nan",
                UserWarning,
            )
    return shifted_spec


def calc_gradient(
    spectra: np.ndarray,
    labels: pd.DataFrame,
    symmetric: bool = True,
    ref_included: bool = True,
) -> pd.DataFrame:
    """
    ToDo: Unit Tests
    Calculates partial derivatives of spectra wrt to each label
    :param np.ndarray spectra: input spectra array
    :param pd.DataFrame labels: input label array
    :param bool symmetric: If true, calculates symmetric gradient about reference
    :param bool ref_included: Is spectra[0] the reference spectrum?
    :return pd.DataFrame: Partial derivatives of spectra wrt each label
    """
    if not isinstance(spectra, np.ndarray):
        raise TypeError("spectra must be np.ndarray")
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be pd.DataFrame")
    nspectra = spectra.shape[0]
    nlabels = labels.shape[0]
    if ref_included:
        skip = 1
    else:
        skip = 0
    if symmetric:
        if nspectra - skip != 2 * nlabels:
            raise ValueError(
                f"nspectra({nspectra-skip}) != 2*nlabel({2*nlabels})"
                + "\nCannot perform symmetric gradient calculation"
            )
        dx = np.diag(labels.iloc[:, skip::2].values - labels.iloc[:, (skip+1)::2].values).copy()
        grad = spectra[skip::2] - spectra[(skip+1)::2]
    else:
        if not ref_included:
            raise ValueError(
                f"Reference Spectra must be included at index 0 "
                + "to calculate asymmetric gradients"
            )
        if nspectra - 1 == nlabels:
            dx = np.diag(labels.iloc[:, 0].values - labels.iloc[:, 1:].values).copy()
            grad = spectra[0] - spectra[1:]
        elif nspectra - 1 == 2 * nlabels:
            dx = np.diag(labels.iloc[:, 0].values - labels.iloc[:, 1::2].values).copy()
            grad = spectra[0] - spectra[1::2]
        else:
            raise ValueError(
                f"nspectra({nspectra - 1}) != nlabel({nlabels}) "
                + f"or 2*nlabel({2 * nlabels})"
                + "\nCannot perform asymmetric gradient calculation"
            )
    dx[labels.index == "Teff"] /= 100  # Scaling dX_Teff
    dx[labels.index == "rv"] /= 10
    dx[dx == 0] = -np.inf
    return pd.DataFrame(grad / dx[:, np.newaxis], index=labels.index)


def kpc_to_mu(
    d: Union[float, List[float], np.ndarray]
) -> Union[np.float64, np.ndarray, Any]:
    """
    Converts kpc to distance modulus
    :param Union[float, List[float], np.ndarray] d: distance in kpc
    :return Union[np.float64, np.ndarray, Any]: distance modulus
    """
    if not isinstance(d, (int, float, np.ndarray, list)):
        raise TypeError("d must be int, float, or np.ndarray/list of floats")
    if isinstance(d, list):
        d = np.array(d)
        d = cast(np.ndarray, d)
    if np.any(d <= 0):
        raise ValueError("d must be > 0")
    return 5 * np.log10(d * 1e3 / 10)


def mu_to_kpc(
    mu: Union[float, List[float], np.ndarray]
) -> Union[float, np.float64, np.ndarray]:
    """
    Converts distance modulus to kpc
    :param Union[float, List[float], np.ndarray] mu: distance modulus
    :return Union[float, np.float64, np.ndarray]: distance in kpc
    """
    if not isinstance(mu, (int, float, np.ndarray, list)):
        raise TypeError("mu must be int, float, or np.ndarray/list of floats")
    if isinstance(mu, list):
        mu = np.array(mu)
        mu = cast(np.ndarray, mu)
    return 1 / (1e3 / 10) * 10 ** (mu / 5)


def download_package_files(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        chunk_size = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size)):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)


def download_all_files(overwrite=True):
    from chemicalc.reference_spectra import label_file, label_id
    from chemicalc.reference_spectra import (
        precomputed_res,
        precomputed_ref_id,
        precomputed_cont_id,
    )

    if label_file.exists() and not overwrite:
        print(f"{label_file} exists")
    else:
        print(f"Downloading {label_file}.h5")
        download_package_files(id=label_id, destination=label_file)

    for res in precomputed_res:
        resolution = precomputed_res[res]
        reference_file = data_dir.joinpath(f"reference_spectra_{resolution:06}.h5")
        reference_id = precomputed_ref_id[res]
        if reference_file.exists() and not overwrite:
            print(f"{reference_file} exists")
        else:
            print(f"Downloading {reference_file}")
            download_package_files(id=reference_id, destination=reference_file)

    for res in precomputed_res:
        resolution = precomputed_res[res]
        continuum_file = data_dir.joinpath(f"reference_continuum_{resolution:06}.h5")
        continuum_id = precomputed_cont_id[res]
        if continuum_file.exists() and not overwrite:
            print(f"{continuum_file} exists")
        else:
            print(f"Downloading {continuum_file}")
            download_package_files(id=continuum_id, destination=continuum_file)


def check_label_format():
    raise NotImplementedError("Coming soon!")


def check_spec_format():
    raise NotImplementedError("Coming soon!")
