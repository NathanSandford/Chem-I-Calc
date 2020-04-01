from typing import Dict, List, Union, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from mendeleev import element
from chemicalc.utils import (
    data_dir,
    doppler_shift,
    convolve_spec,
    calc_gradient,
)
from chemicalc.utils import download_package_files


precomputed_res: Dict[str, float] = {
    "max": 300000,
    #'high': 100000,
    #'med': 50000,
    #'low': 25000
}
precomputed_ref_id: Dict[str, str] = {"max": "1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2"}
precomputed_cont_id: Dict[str, str] = {"max": "1Fhx1KM8b6prtCGOZ3NazVeDQY-x9gOOU"}

label_file: Path = data_dir.joinpath("reference_labels.h5")
label_id: str = "1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor"
if not label_file.exists():
    print("Downloading label_file")
    download_package_files(id=label_id, destination=label_file)

reference_stars: List[str] = list(pd.read_hdf(label_file, "ref_list").values.flatten())
elements_included: List[str] = [x.symbol for x in element(list(range(3, 100)))]
alpha_el: List[str] = ["O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Ti"]


class ReferenceSpectra:
    def __init__(
        self,
        reference: str,
        normalized: bool = True,
        res: str = "max",
        iron_scaled: bool = False,
        alpha_included: bool = False,
        radius: float = 1,
        dist: float = 10,
    ) -> None:
        """

        :param str reference:
        :param bool normalized:
        :param str res:
        :param bool iron_scaled:
        :param bool alpha_included:
        :param float radius:
        :param float dist:
        """
        self.reference = reference
        self.resolution = dict(init=precomputed_res[res])
        self.reference_file = data_dir.joinpath(
            f'reference_spectra_{self.resolution["init"]:06}.h5'
        )
        if not self.reference_file.exists():
            print(
                "Downloading reference file---this may take a few minutes but is only necessary once"
            )
            download_package_files(
                id=precomputed_ref_id[res], destination=self.reference_file
            )
        wave_df = pd.read_hdf(self.reference_file, "highres_wavelength")
        spec_df = pd.read_hdf(self.reference_file, reference)
        if not normalized:
            self.continuum_file = data_dir.joinpath(
                f'reference_continuum_{self.resolution["init"]:06}.h5'
            )
            if not self.continuum_file.exists():
                print(
                    "Downloading continuum file---this may take a few minutes but is only necessary once"
                )
                download_package_files(
                    id=precomputed_cont_id[res], destination=self.continuum_file
                )
            cont_df = pd.read_hdf(self.continuum_file, reference)
            spec_df *= cont_df
            spec_df = calc_f_nu(spectra=spec_df, radius=radius, dist=dist)
        label_df = pd.read_hdf(label_file, reference)
        if not iron_scaled:
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

    def add_rv_spec(self, d_rv: float) -> None:
        """
        ToDo: DocString
        :param float d_rv:
        :return:
        """
        self.labels.loc["RV"] = 0.0
        self.labels["fffff"] = self.labels["aaaaa"]
        self.labels["ggggg"] = self.labels["aaaaa"]
        self.labels.loc["RV", "fffff"] += d_rv
        self.labels.loc["RV", "ggggg"] -= d_rv
        tmp1 = doppler_shift(self.wavelength["init"], self.spectra["init"][0], d_rv)
        tmp2 = doppler_shift(self.wavelength["init"], self.spectra["init"][0], -d_rv)
        self.spectra["init"] = np.append(
            self.spectra["init"], tmp1[np.newaxis, :], axis=0
        )
        self.spectra["init"] = np.append(
            self.spectra["init"], tmp2[np.newaxis, :], axis=0
        )

    def convolve(self, instrument, name: str = None) -> None:
        """

        :param instrument:
        :param str name:
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

    def calc_synth_phot(
        self, filter_set, name: Optional[str] = None, spectrum_name="init"
    ) -> None:
        """

        :param filter_set:
        :param name:
        :param str spectrum_name:
        :return:
        """
        if name is None:
            name = filter_set.name
        if spectrum_name != "init":
            filter_interp = interp1d(
                filter_set.throughput.index.values,
                filter_set.throughput.values,
                axis=0,
                fill_value=0,
                bounds_error=False,
            )
            throughput = filter_interp(self.wavelength[spectrum_name])
        else:
            throughput = filter_set.throughput.values
        self.spectra[name] = calc_MagAB(
            f_nu=self.spectra[spectrum_name],
            throughput=throughput,
            wave=self.wavelength[spectrum_name],
        )
        self.wavelength[name] = np.array(list(filter_set.wave_eff.values()))
        self.filters[name] = list(filter_set.throughput.columns)
        self.resolution[name] = 0

    def calc_gradient(
        self,
        name: str,
        symmetric: bool = True,
        ref_included: bool = True,
        v_micro_scaling: float = 1,
        d_rv: bool = None,
    ) -> None:
        """

        :param name:
        :param symmetric:
        :param ref_included:
        :param v_micro_scaling:
        :param d_rv:
        :return:
        """
        self.gradients[name] = calc_gradient(
            self.wavelength[name],
            self.spectra[name],
            self.labels,
            symmetric=symmetric,
            ref_included=ref_included,
            v_micro_scaling=v_micro_scaling,
            d_rv=d_rv,
        )
        self.gradients[name].columns = self.wavelength[name]

    def zero_gradients(self, name: str, labels: List[str]):
        """

        :param name:
        :param labels:
        :return:
        """
        self.gradients[name].loc[labels] = 0

    def get_names(self) -> List[str]:
        """

        :return:
        """
        return list(self.spectra.keys())

    def reset(self) -> None:
        """

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
