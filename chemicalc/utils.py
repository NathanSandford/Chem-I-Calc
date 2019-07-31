import os
from pathlib import Path
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
from numpy.fft import rfftfreq
from scipy.interpolate import interp1d
from chemicalc import exception as e

data_dir = Path(os.path.dirname(__file__)).joinpath('data')
data_dir.mkdir(exist_ok=True)


def generate_wavelength_template(start_wavelength: float, end_wavelength: float,
                                 resolution: float, truncate: bool = False):
    """

    :param start_wavelength:
    :param end_wavelength:
    :param resolution:
    :param truncate:
    :return:
    """
    '''
    Generate wavelength array with fixed resolution.

    args:
    start_wavelength = minimum wavelength of spectra to include
    end_wavelength = maximum wavelength of spectra to include
    resolution = resolving power of instrument (R = lambda / delta lambda)
    truncate = Boolean, if true, drop final pixel for which lambda > end_wavelength

    returns = wavelength grid of given resolution between start and end wavelengths
    '''
    # TODO: generate_wavelength_template doc-string
    wavelength_template = [start_wavelength]
    wavelength_now = start_wavelength

    while wavelength_now < end_wavelength:
        wavelength_now += wavelength_now / resolution
        wavelength_template.append(wavelength_now)
    wavelength_template = np.array(wavelength_template)

    if truncate:
        wavelength_template = wavelength_template[:-1]
    return wavelength_template


def convolve_spec(wave, spec, resolution, outwave, res_in=None):
    """

    :param wave:
    :param spec:
    :param resolution:
    :param outwave:
    :param res_in:
    :return:
    """
    # TODO: convolve_spec doc-string
    sigma_to_fwhm = 2.355

    width = resolution * sigma_to_fwhm
    sigma_out = (resolution * sigma_to_fwhm) ** -1
    if res_in is None:
        sigma_in = 0
    else:
        sigma_in = (res_in * sigma_to_fwhm) ** -1

    # Trim Wavelength Range
    nsigma_pad = 20.0
    wlim = np.array([outwave.min(), outwave.max()])
    wlim *= (1 + nsigma_pad / width * np.array([-1, 1]))
    mask = (wave > wlim[0]) & (wave < wlim[1])
    wave = wave[mask]
    if spec.ndim == 1:
        spec = spec[mask]
    else:
        spec = spec[:, mask]

    # Make Convolution Grid
    wmin, wmax = wave.min(), wave.max()
    nwave = wave.shape[0]
    nwave_new = int(2 ** (np.ceil(np.log2(nwave))))
    lnwave_new = np.linspace(np.log(wmin), np.log(wmax), nwave_new)
    wave_new = np.exp(lnwave_new)
    fspec = interp1d(wave, spec,
                     bounds_error=False, fill_value='extrapolate')
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
    fspec = interp1d(wave, spec_conv,
                     bounds_error=False, fill_value='extrapolate')
    return fspec(outwave)


def calc_gradient(spectra, labels, symmetric=True, ref_included=True, v_micro_scaling=1e5):
    """

    :param spectra:
    :param labels:
    :param symmetric:
    :param ref_included:
    :param v_micro_scaling:
    :return:
    """
    # TODO: calc_gradient doc-string
    nspectra = spectra.shape[0]
    nlabels = labels.shape[0]
    if ref_included:
        skip = 1
    else:
        skip = 0
    if symmetric:
        if nspectra-skip != 2*nlabels:
            raise e.GradientError(f"nspectra({nspectra-skip}) != 2*nlabel({2*nlabels})"
                                  + "\nCannot perform symmetric gradient calculation")
        dx = np.diag(labels.iloc[:, 1::2].values - labels.iloc[:, 2::2].values).copy()
        grad = spectra[1::2] - spectra[2::2]
    else:
        if not ref_included:
            raise e.GradientError(f"Reference Spectra must be included at index 0"
                                  + "to calculate asymmetric gradients")
        if nspectra - 1 == nlabels:
            dx = np.diag(labels.iloc[:, 0].values - labels.iloc[:, 1:].values).copy()
            grad = spectra[0] - spectra[1:]
        elif nspectra - 1 == 2 * nlabels:
            dx = np.diag(labels.iloc[:, 0].values - labels.iloc[:, 1::2].values).copy()
            grad = spectra[0] - spectra[1::2]
        else:
            raise e.GradientError(f"nspectra({nspectra - 1}) != nlabel({nlabels})"
                                  + f"or 2*nlabel({2 * nlabels})"
                                  + "\nCannot perform asymmetric gradient calculation")
    dx[0] /= 100  # Scaling dX_Teff
    dx[2] /= v_micro_scaling  # Scaling dX_v_micro
    dx[dx == 0] = -np.inf
    return pd.DataFrame(grad / dx[:, np.newaxis], index=labels.index)


def calc_crlb(reference, instruments, priors=None, output_fisher=False):
    """

    :param reference:
    :param instruments:
    :param priors:
    :param output_fisher:
    :return:
    """
    # TODO: calc_crlb doc-string
    if type(instruments) == list:
        grad_list = []
        snr_list = []
        for instrument in instruments:
            grad_list.append(reference.gradients[instrument.name].values)
            snr_list.append(instrument.snr)
        grad = np.concatenate(grad_list, axis=1)
        snr2 = np.diag(np.concatenate(snr_list, axis=0))**2
    else:
        grad = reference.gradients[instruments.name].values
        snr2 = np.diag(instruments.snr)**2

    fisher_mat = (grad.dot(snr2)).dot(grad.T)
    diag_val = (np.abs(np.diag(fisher_mat)) < 1.)
    fisher_mat[diag_val, :] = 0.
    fisher_mat[:, diag_val] = 0.
    fisher_mat[diag_val, diag_val] = 10. ** -6
    fisher_df = pd.DataFrame(fisher_mat, columns=reference.labels.index, index=reference.labels.index)

    if priors:
        for label in priors:
            prior = priors[label]
            if prior is None:
                continue
            if label == 'Teff':
                prior /= 100
            if prior == 0:
                fisher_df.loc[label, :] = 0
                fisher_df.loc[:, label] = 0
                fisher_df.loc[label, label] = 1e-6
            else:
                fisher_df.loc[label, label] += prior**(-2)

    crlb = pd.DataFrame(np.sqrt(np.diag(np.linalg.pinv(fisher_df))), index=reference.labels.index)

    if output_fisher:
        return crlb, fisher_df
    else:
        return crlb


def sort_crlb(crlb, cutoff, sort_by='default'):
    """

    :param crlb:
    :param cutoff:
    :param sort_by:
    :return:
    """
    crlb_temp = crlb[:3].copy()
    crlb[crlb > cutoff] = np.NaN
    crlb[:3] = crlb_temp

    if sort_by == 'default':
        sort_by_index = np.sum(pd.isna(crlb)).idxmin()
    else:
        if sort_by == list(crlb.columns):
            sort_by_index = sort_by
        else:
            assert False, f"{sort_by} not in CR_Gradients_File"

    valid_ele \
        = np.concatenate(
        [crlb.index[:3],
         crlb.index[3:][np.min(crlb[3:], axis=1) < cutoff]])
    valid_ele_sorted \
        = np.concatenate(
        [crlb.index[:3],
         crlb.loc[valid_ele][3:].sort_values(sort_by_index).index])

    crlb = crlb.loc[valid_ele_sorted]
    return crlb


def kpc_to_mu(d: float):
    """

    :param d: distance in kpc
    :return: distance modulus
    """
    return 5 * np.log10(d * 1e3 / 10)


def download_package_files(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        chunk_size = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)