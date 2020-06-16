from typing import Any, List, Tuple, Dict, Union, Optional, cast
from warnings import warn
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg
from chemicalc.reference_spectra import ReferenceSpectra, alpha_el
from chemicalc.instruments import InstConfig


def init_crlb_df(reference: ReferenceSpectra) -> pd.DataFrame:
    """
    Initialized CRLB dataframe with indices corresponding to all the labels included

    :param ReferenceSpectra reference: Reference star object (used to identify stellar labels)
    :return pd.DataFrame: Empty CRLB dataframe
    """
    if not isinstance(reference, ReferenceSpectra):
        raise TypeError(
            "reference must be a chemicalc.reference_spectra.ReferenceSpectra object"
        )
    return pd.DataFrame(index=reference.labels.index)


def calc_crlb(
    reference: ReferenceSpectra,
    instruments: Union[InstConfig, List[InstConfig]],
    pixel_corr: Optional[List[float]] = None,
    priors: Optional[Dict["str", float]] = None,
    bias_grad: Optional[pd.DataFrame] = None,
    use_alpha: bool = False,
    output_fisher: bool = False,
    chunk_size: int = 10000,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculates the Fisher Information Matrix and Cramer-Rao Lower Bound from spectral gradients

    :param ReferenceSpectra reference: Reference star object
    :param Union[InstConfig,List[InstConfig]] instruments: Instrument object or list of instrument objects
    :param Optional[List[float]] pixel_corr: Correlation of adjacent pixels.
                                             This may considerably slow down the computation.
    :param Optional[Dict[str,float]] priors: 1-sigma Gaussian priors for labels
    :param Optional[pd.DataFrame] bias_grad: Gradient of the bias matrix
    :param bool use_alpha: If true, uses bulk alpha gradients and zeros gradients of individual alpha elements
                           (see chemicalc.reference_spectra.alpha_el)
    :param bool output_fisher: If true, outputs Fisher information matrix
    :param int chunk_size: Number of pixels to break spectra into. Helps with memory usage for large spectra.
    :return Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]: DataFrame of CRLBs.
                                                                  If output_fisher=True, also returns FIM
    """
    if not isinstance(reference, ReferenceSpectra):
        raise TypeError(
            "reference must be a chemicalc.reference_spectra.ReferenceSpectra object"
        )
    if not isinstance(instruments, list):
        instruments = [instruments]
    if chunk_size is not None:
        if chunk_size < 1000:
            warn(
                f"chunk_size of {chunk_size} seems a little small...This may lead to numerical errors",
                UserWarning,
            )
    fisher_mat = np.zeros((reference.nlabels, reference.nlabels))
    for instrument in instruments:
        if not isinstance(instrument, InstConfig):
            raise TypeError(
                "instruments must be chemicalc.instruments.InstConfig objects"
            )
        if instrument.name not in reference.gradients.keys():
            raise KeyError(
                f"Reference star does not have gradients for {instrument.name}"
            )
        grad_backup = reference.gradients[instrument.name].copy()
        if use_alpha and "alpha" not in reference.labels.index:
            raise ValueError("alpha not included in reference file")
        elif use_alpha:
            reference.zero_gradients(name=instrument.name, labels=alpha_el)
        elif not use_alpha and "alpha" in reference.labels.index:
            reference.zero_gradients(name=instrument.name, labels=["alpha"])
        grad = reference.gradients[instrument.name].values
        reference.gradients[instrument.name] = grad_backup
        flux_var = instrument.snr ** (-2)
        if chunk_size is not None:
            n_chunks = int(np.ceil(grad.shape[1] / chunk_size))
        else:
            chunk_size = grad.shape[1]
            n_chunks = 1
        for i in range(n_chunks):
            grad_tmp = grad[:, i * chunk_size : (i + 1) * chunk_size]
            flux_var_tmp = flux_var[i * chunk_size : (i + 1) * chunk_size]
            if pixel_corr:
                flux_covar = sparse.diags(flux_var_tmp, format="csc")
                for k, covar_factor in enumerate(pixel_corr):
                    j = k + 1
                    flux_covar += covar_factor * sparse.diags(
                        flux_var_tmp[:-j], j
                    ) + covar_factor * sparse.diags(flux_var_tmp[j:], -j)
                flux_covar_inv = linalg.inv(flux_covar).todense()
            else:
                flux_covar_inv = np.diag(flux_var_tmp ** -1)
            fisher_mat += grad_tmp.dot(flux_covar_inv).dot(grad_tmp.T)
    diag_val = np.abs(np.diag(fisher_mat)) < 1.0
    fisher_mat[diag_val, :] = 0.0
    fisher_mat[:, diag_val] = 0.0
    fisher_mat[diag_val, diag_val] = 10.0 ** -6
    fisher_df = pd.DataFrame(
        fisher_mat, columns=reference.labels.index, index=reference.labels.index
    )
    if priors:
        if not isinstance(priors, dict):
            raise TypeError("priors must be None or a dictionary of {label: prior}")
        for label in priors:
            prior = priors[label]
            if not isinstance(prior, (int, float)):
                raise TypeError("prior dict entries must be int, or float")
            if label not in reference.labels.index:
                raise KeyError(f"{label} is not included in reference")
            if prior is None:
                continue
            if label == "Teff":
                prior /= 100
            if prior == 0:
                fisher_df.loc[label, :] = 0
                fisher_df.loc[:, label] = 0
                fisher_df.loc[label, label] = 1e-6
            else:
                fisher_df.loc[label, label] += prior ** (-2)
    if bias_grad is not None:
        warn(
            "Calculating the biased CRLB is an experimental feature and has not been thoroughly tested.",
            UserWarning,
        )
        I = np.eye(fisher_df.shape[0])
        D = bias_grad
        crlb = pd.DataFrame(
            np.sqrt(np.diag(((I + D).dot(np.linalg.pinv(fisher_df))).dot((I + D).T))),
            index=reference.labels.index,
        )
    else:
        crlb = pd.DataFrame(
            np.sqrt(np.diag(np.linalg.pinv(fisher_df))), index=reference.labels.index
        )
    if output_fisher:
        return crlb, fisher_df
    else:
        return crlb


def sort_crlb(
    crlb: pd.DataFrame,
    cutoff: float,
    sort_by: str = "default",
    fancy_labels: bool = False,
) -> pd.DataFrame:
    """
    Sorts CRLB dataframe by decreasing precision of labels and removes labels with precisions worse than cutoff.

    :param pd.DataFrame crlb: dataframe of CRLBs
    :param float cutoff: Cutoff precision of labels
    :param str sort_by: Name of dataframe column to sort labels by.
                        Default uses the column with the most labels recovered below the cutoff
    :param bool fancy_labels: Replaces Teff, logg, and v_micro with math-formatted labels (for plotting).
    :return pd.DataFrame: Sorted CRLB dataframe
    """
    # ToDo: Thoroughly check that sorting works as expected
    if not isinstance(crlb, pd.DataFrame):
        raise TypeError("crlb must be pd.DataFrame")
    if not isinstance(cutoff, (int, float)):
        raise TypeError("cutoff must be int or float")
    if not isinstance(sort_by, str):
        raise TypeError(f"sort_by must be str in {list(crlb.columns)} or 'default'")
    crlb_temp = crlb[:3].copy()
    crlb[crlb > cutoff] = np.NaN
    crlb[:3] = crlb_temp
    if sort_by == "default":
        sort_by_index = np.sum(pd.isna(crlb)).idxmin()
    else:
        if sort_by in list(crlb.columns):
            sort_by_index = sort_by
        else:
            raise KeyError(
                f"{sort_by} not in crlb \n Try 'default' or one of {list(crlb.columns)}"
            )
    valid_ele = np.concatenate(
        [crlb.index[:3], crlb.index[3:][np.min(crlb[3:], axis=1) < cutoff]]
    )
    valid_ele_sorted = np.concatenate(
        [crlb.index[:3], crlb.loc[valid_ele][3:].sort_values(sort_by_index).index]
    )
    crlb = crlb.loc[valid_ele_sorted]
    if len(crlb.index) == 3:
        warn(f"No elements w/ CRLBs < cutoff ({cutoff})", UserWarning)
    if fancy_labels:
        crlb.index = [r"$T_{eff}$ (100 K)", r"$\log(g)$", r"$v_{micro}$ (km/s)"] + list(
            crlb.index[3:]
        )
    return crlb
