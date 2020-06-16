from typing import Dict, List, Union, Optional, Tuple
from warnings import warn
from pathlib import Path
import pandas as pd
import numpy as np
from mendeleev import element
from chemicalc.instruments import InstConfig
from chemicalc.utils import (
    doppler_shift,
    convolve_spec,
    calc_gradient,
)
from chemicalc.file_mgmt import (
    data_dir,
    download_package_files,
    precomputed_res,
    precomputed_ref_id,
    precomputed_label_id,
    precomputed_alpha_included,
)

# noinspection PyTypeChecker
elements_included: List[str] = [x.symbol for x in element(list(range(3, 100)))]
"""
List[str]: List of all elements included in the pre-computed spectral grids.
"""
alpha_el: List[str] = ["O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti"]
"""
List[str]: List of elements considered when calculating bulk [:math:`\\alpha`/H].
"""


class ReferenceSpectra:
    """
    Object for spectra of a specific reference star

    :param str reference: Name of reference star to load (e.g., 'RGB_m1.5')
    :param str init_res: Initial resolution of high-res reference spectra.
                         Only 300000 is presently included for default spectra.
                         Can be approximate if using custom reference spectra.
    :param bool scale_by_iron: If true, scales all elemental abundances by [Fe/H]
    :param bool alpha_included: If true, will include an alpha label after the atmospheric parameters
                                and before the other elements (i.e., between v_micro and Li)
    :param \**kwargs: see below

    :keyword str ref_spec_file: Full path to file of reference spectra
    :keyword str ref_label_file: Full path to file of reference spectra labels

    :ivar Dict[Union[str,float],int] resolution: Dictionary of resolving powers for each instrument
    :ivar ref_spec_file: Full path to file of reference spectra
    :ivar ref_label_file: Full path to file of reference spectra labels
    :ivar Dict[Union[str,float],np.ndarray] wavelength: Dictionary of wavelength arrays for each instrument
    :ivar Dict[Union[str,float],np.ndarray] spectra: Dictionary of spectral grids for each instrument
    :ivar pd.DataFrame labels: Labels corresponding to each spectrum in the grid
    :ivar Dict[Union[str,float],pd.DataFrame] gradients: Dictionary of partial derivatives for each instrument
    :ivar int nspectra: Number of spectra included in the grid
    :ivar int nlabels: Number of labels included
    """

    def __init__(
        self,
        reference: str,
        init_res: float = precomputed_res[0],
        scale_by_iron: bool = False,
        alpha_included: bool = True,
        **kwargs,
    ) -> None:
        if not isinstance(reference, str):
            raise TypeError("reference must be str")
        if not isinstance(init_res, (int, float)):
            raise TypeError("init_res must be float")
        self.reference = reference
        self.resolution = {"init": init_res}

        if "ref_spec_file" in kwargs:
            self.ref_spec_file = Path(kwargs["ref_spec_file"])
            if not self.ref_spec_file.exists():
                raise ValueError(f"ref_spec_file {self.ref_spec_file} does not exist")
        else:
            if not self.resolution["init"] in precomputed_res:
                raise ValueError(f"{init_res} not a precomputed resolution")
            self.ref_spec_file = data_dir.joinpath(
                f"reference_spectra_{init_res:06}.h5"
            )
            if not self.ref_spec_file.exists():
                print(
                    "Downloading reference file---this may take a few minutes but is only necessary once"
                )
                download_package_files(
                    id_str=precomputed_ref_id[init_res], destination=self.ref_spec_file
                )

        if "ref_label_file" in kwargs:
            self.ref_label_file = Path(kwargs["ref_label_file"])
            if not self.ref_label_file.exists():
                raise ValueError(f"ref_label_file {self.ref_label_file} does not exist")
        else:
            self.ref_label_file = data_dir.joinpath("reference_labels.h5")
            if not self.ref_label_file.exists():
                print(
                    "Downloading label_file---this should be quick and is only necessary once"
                )
                download_package_files(
                    id_str=precomputed_label_id, destination=self.ref_label_file
                )

        ref_list_spec = list(
            pd.DataFrame(pd.read_hdf(self.ref_spec_file, "ref_list")).values.flatten()
        )
        ref_list_label = list(
            pd.DataFrame(pd.read_hdf(self.ref_label_file, "ref_list")).values.flatten()
        )
        if not (reference in ref_list_spec) and (reference in ref_list_label):
            raise ValueError(
                f"{reference} is not included in ref_label_file and/or ref_spec_file"
            )
        else:
            if alpha_included and reference not in precomputed_alpha_included:
                raise ValueError(
                    f"alpha offsets not currently included for {reference}"
                )
        wave_df = pd.DataFrame(pd.read_hdf(self.ref_spec_file, "highres_wavelength"))
        spec_df = pd.DataFrame(pd.read_hdf(self.ref_spec_file, reference))
        label_df = pd.DataFrame(pd.read_hdf(self.ref_label_file, reference))
        if scale_by_iron:
            label_df.loc[set(elements_included) ^ {"Fe"}] -= label_df.loc["Fe"]
        if alpha_included:
            label_df = pd.concat(
                [
                    label_df.iloc[:3],
                    pd.DataFrame(label_df.loc[alpha_el].mean()).T,
                    label_df.iloc[3:],
                ]
            )
            label_df.index = ["Teff", "logg", "v_micro", "alpha"] + elements_included
            if (
                np.abs(
                    label_df.loc["alpha"][[4, 6, 7]].max() - label_df.loc["alpha"][0]
                )
                < 0.001
            ):
                warn(
                    "Expected offset in alpha not found. Are you sure this reference spectra includes alpha gradients?"
                    + "\nIf so, they must come immediately after v_micro offsets in both the label and spectra files.",
                    UserWarning,
                )
        else:
            label_df.index = ["Teff", "logg", "v_micro"] + elements_included

        self.wavelength = dict(init=wave_df.to_numpy().T[0])
        self.spectra = dict(init=spec_df.to_numpy().T)
        self.labels = label_df
        self.gradients: Dict[Union[str, float], pd.DataFrame] = {}

        self.nspectra = self.spectra["init"].shape[0]
        self.nlabels = self.labels.shape[0]

    def add_rv_spec(self, d_rv: float, symmetric: bool = True) -> None:
        """
        Adds spectra and labels corresponding to a small doppler shift of the reference spectra.
        Assumes that the first spectra in ref_spec_file is a reference w/ no offsets to any labels.

        :param float d_rv: small doppler shift in km/s
        :param bool symmetric: if True, applies both positive and negative doppler shifts
        :return:
        """
        warn(
            "This feature is experimental and has not been sufficiently tested on either computational or "
            "statistical grounds!",
            UserWarning,
        )
        self.labels.loc["RV"] = 0.0
        self.labels["fffff"] = self.labels["aaaaa"]
        self.labels.loc["RV", "fffff"] += d_rv
        tmp1 = doppler_shift(self.wavelength["init"], self.spectra["init"][0], d_rv)
        self.spectra["init"] = np.append(
            self.spectra["init"], tmp1[np.newaxis, :], axis=0
        )
        if symmetric:
            self.labels["ggggg"] = self.labels["aaaaa"]
            self.labels.loc["RV", "ggggg"] -= d_rv
            tmp2 = doppler_shift(
                self.wavelength["init"], self.spectra["init"][0], -d_rv
            )
            self.spectra["init"] = np.append(
                self.spectra["init"], tmp2[np.newaxis, :], axis=0
            )

    def convolve(self, instrument, name: Optional[str] = None) -> None:
        """
        Convolves spectra to instrument resolution and samples onto instrument's wavelength grid

        :param InstConfig instrument: Instrument object to convolve and sample spectra onto
        :param str name: Name to give spectra. If None, defaults to name of instrument
        :return:
        """
        if name is None:
            name = instrument.name
        outwave = instrument.wave
        self.spectra[name] = convolve_spec(
            wave=self.wavelength["init"],
            spec=self.spectra["init"],
            resolution=instrument.R_res,
            outwave=outwave,
            res_in=self.resolution["init"],
        )
        self.wavelength[name] = outwave
        self.resolution[name] = instrument.R_res

    def calc_gradient(
        self,
        name: Union[str, InstConfig],
        symmetric: bool = True,
        ref_included: bool = True,
    ) -> None:
        """
        Calculates gradients of the reference spectra with respect to each label.

        :param Union[str,InstConfig] name: Name of convolved spectra to calculate gradient for.
            Will also accept an InstConfig object and use InstConfig.name.
        :param bool symmetric: If True, calculates symmetric gradient around reference labels
        :param bool ref_included: If True, expects first spectra to be reference spectra w/ no offsets to any labels.
                                  Required for symmetric=False.
        :return:
        """
        if isinstance(name, InstConfig):
            name = name.name

        self.gradients[name] = calc_gradient(
            spectra=self.spectra[name],
            labels=self.labels,
            symmetric=symmetric,
            ref_included=ref_included,
        )
        self.gradients[name].columns = self.wavelength[name]

    def zero_gradients(
        self, name: Union[str, InstConfig], labels: Union[str, List[str]]
    ):
        """
        Sets gradients of a spectrum to zero for the specified labels. This is equivalent to setting a delta-function
        prior on those labels (i.e., holding them fixed).

        :param Union[str,InstConfig] name: Name of spectra to apply gradient zeroing to.
            Will also accept an InstConfig object and use InstConfig.name.
        :param Union[str,List[str]] labels: List of labels for which to zero gradients
        :return:
        """
        if isinstance(name, InstConfig):
            name = name.name
        self.gradients[name].loc[labels] = 0

    def mask_wavelength(
        self, name: Union[str, InstConfig], regions: List[Tuple[float, float]]
    ) -> None:
        """
        Masks the information content of a spectrum by setting the gradient to zero within the bounds of the mask.
        Can be used to mimic the masking of skylines, non-LTE lines, or detector gaps.

        :param Union[str,InstConfig] name: Name of the spectra to apply  mask to
        :param List[Tuple[float,float]] regions: List of wavelength bounds on the regions to mask.
        :return:
        """
        if isinstance(name, InstConfig):
            name = name.name
        if not isinstance(regions, list):
            regions = [regions]
        for region in regions:
            min_wave, max_wave = region
            mask = (self.wavelength[name] > min_wave) & (
                self.wavelength[name] < max_wave
            )
            self.gradients[name].iloc[:, mask] = 0

    def get_names(self) -> List[str]:
        """
        Get names of all spectra contained in this object

        :return List[str]: List of spectra names that this object contains.
        """
        return list(self.spectra.keys())

    def duplicate(self, name: str, new_name: str) -> None:
        """
        Duplicates set of spectra/gradients

        :param str name: Name of spectra to duplicate
        :param str new_name: Name given to new spectra
        :return:
        """
        self.resolution[new_name] = self.resolution[name]
        self.wavelength[new_name] = np.copy(self.wavelength[name])
        self.spectra[new_name] = np.copy(self.spectra[name])
        self.gradients[new_name] = pd.DataFrame.copy(self.gradients[name])

    def reset(self) -> None:
        """
        Resets object to only the initial high-res spectra

        :return:
        """
        init_resolution = self.resolution["init"]
        self.resolution.clear()
        self.resolution["init"] = init_resolution

        init_wavelength = self.wavelength["init"]
        self.wavelength.clear()
        self.wavelength["init"] = init_wavelength

        init_spectra = self.spectra["init"]
        self.spectra.clear()
        self.spectra["init"] = init_spectra
        self.spectra["init"] = init_spectra

        del self.gradients
        self.gradients = {}
