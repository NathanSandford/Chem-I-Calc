from typing import Dict, List, Union, Optional, TYPE_CHECKING
from pathlib import Path
import pandas as pd
import numpy as np
from mendeleev import element
from chemicalc.utils import (
    data_dir,
    doppler_shift,
    convolve_spec,
    calc_gradient,
    download_package_files,
)


precomputed_res: List = [300000]
precomputed_ref_id: Dict[float, str] = {300000: "1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2"}
precomputed_cont_id: Dict[float, str] = {300000: "1Fhx1KM8b6prtCGOZ3NazVeDQY-x9gOOU"}

label_id: str = "1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor"

elements_included: List[str] = [x.symbol for x in element(list(range(3, 100)))]
alpha_el: List[str] = ["O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti"]


class ReferenceSpectra:
    def __init__(
        self,
        reference: str,
        init_res: float = precomputed_res[0],
        scale_by_iron: bool = True,
        alpha_included: bool = False,
        **kwargs
    ) -> None:
        """
        ToDo: Unit Tests
        Object for spectra of a specific reference star
        :param str reference: Name of reference star to load (e.g., 'RGB_m1.5')
        :param str init_res: Initial resolution of high-res reference spectra. Only 300000 is presently included for default spectra. Can be approximate if using custom reference spectra.
        :param bool scale_by_iron: If true, scales all elemental abundances by [Fe/H]
        :param bool alpha_included: If true, will include an alpha label after the atmospheric parameters and before the other elements (i.e., between v_micro and Li)
        :param kwargs: see below

        Keyword Arguments:
        :key str ref_spec_file: Full path to file of reference spectra
        :key str ref_label_file: Full path to file of reference spectra labels
        """
        if not isinstance(reference, str):
            raise TypeError("reference must be str")
        if not isinstance(init_res, (int, float)):
            raise TypeError("init_res must be float")
        self.reference = reference
        self.resolution = {"init": init_res}

        if 'ref_spec_file' in kwargs:
            self.ref_spec_file = Path(kwargs['ref_spec_file'])
            if not self.ref_spec_file.exists():
                raise ValueError(f"ref_spec_file {self.ref_spec_file} does not exist")
        else:
            if not self.resolution["init"] in precomputed_res:
                raise ValueError(f"{self.resolution} not a precomputed resolution")
            self.ref_spec_file = data_dir.joinpath(
                f'reference_spectra_{init_res:06}.h5'
            )
            if not self.ref_spec_file.exists():
                print(
                    "Downloading reference file---this may take a few minutes but is only necessary once"
                )
                download_package_files(
                    id=precomputed_ref_id[init_res], destination=self.ref_spec_file
                )

        if 'ref_label_file' in kwargs:
            self.ref_label_file = Path(kwargs['ref_label_file'])
            if not self.ref_label_file.exists():
                raise ValueError(f"ref_label_file {self.ref_label_file} does not exist")
        else:
            self.ref_label_file = data_dir.joinpath("reference_labels.h5")
            if not self.ref_label_file.exists():
                print("Downloading label_file---this should be quick and is only necessary once")
                download_package_files(id=label_id, destination=self.ref_label_file)

        ref_list_spec = list(pd.DataFrame(pd.read_hdf(self.ref_spec_file, "ref_list")).values.flatten())
        ref_list_label = list(pd.DataFrame(pd.read_hdf(self.ref_label_file, "ref_list")).values.flatten())
        if not (reference in ref_list_spec) and (reference in ref_list_label):
            raise ValueError(f"{reference} is not included in ref_label_file and/or ref_spec_file")

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
        else:
            label_df.index = ["Teff", "logg", "v_micro"] + elements_included

        self.wavelength = dict(init=wave_df.to_numpy().T[0])
        self.spectra = dict(init=spec_df.to_numpy().T)
        self.labels = label_df
        self.gradients: Dict[Union[str, float], pd.DataFrame] = {}
        self.filters: Dict[Union[str, float], List[str]] = {}

        self.nspectra = self.spectra["init"].shape[0]
        self.nlabels = self.labels.shape[0]

    def add_rv_spec(self, d_rv: float, symmetric: bool = True) -> None:
        """
        ToDo: Unit Tests
        Adds spectra and labels corresponding to a small doppler shift of the reference spectra. Assumes that the first spectra in ref_spec_file is a reference w/ no offsets to any labels
        :param float d_rv: small doppler shift in km/s
        :param bool symmetric: if True, applies both positive and negative doppler shifts
        :return:
        """
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
            tmp2 = doppler_shift(self.wavelength["init"], self.spectra["init"][0], -d_rv)
            self.spectra["init"] = np.append(
                self.spectra["init"], tmp2[np.newaxis, :], axis=0
            )

    def convolve(self, instrument, name: Optional[str] = None) -> None:
        """
        ToDo: Unit Tests
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
        name: str,
        symmetric: bool = True,
        ref_included: bool = True,
    ) -> None:
        """
        ToDo: Unit Tests
        Calculates gradients of the reference spectra with respect to each label.
        :param str name: Name of convolved spectra to calculate gradient for
        :param bool symmetric: If True, calculates symmetric gradient around reference labels
        :param bool ref_included: If True, expects first spectra to be reference spectra w/ no offsets to any labels. Required for symmetric=False.
        :return:
        """
        self.gradients[name] = calc_gradient(
            spectra=self.spectra[name],
            labels=self.labels,
            symmetric=symmetric,
            ref_included=ref_included,
        )
        self.gradients[name].columns = self.wavelength[name]

    def zero_gradients(self, name: str, labels: List[str]):
        """
        ToDo: Unit Tests
        :param name: Name of spectra to apply gradient zeroing to
        :param labels: List of labels for which to zero gradients
        :return:
        """
        self.gradients[name].loc[labels] = 0

    def get_names(self) -> List[str]:
        """
        ToDo: Unit Tests
        Get names of all spectra contained in this object
        :return List[str]: List of spectra names that this object contains.
        """
        return list(self.spectra.keys())

    def reset(self) -> None:
        """
        ToDo: Unit Tests
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

        del self.gradients
