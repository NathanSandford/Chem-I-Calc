from typing import Union, Dict
from warnings import warn
import copy
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import interp1d
from chemicalc.utils import generate_wavelength_template
from chemicalc.s2n import Sig2NoiseQuery
from chemicalc.file_mgmt import data_dir, inst_file


sampX: float = 3  # Placeholder for Unknown Wavelength Sampling


class InstConfig:
    """
    Object containing Instrument configuration.

    :param str name: Name of the instrument configuration
    :param float res: Resolving power (:math:`\\lambda/d\\lambda`)
    :param float samp: Pixels per Resolution element
    :param float start: Beginning wavelength in Angstroms
    :param end: Ending wavelength in Angstroms
    :param truncate: If true, discards any pixels with wavelengths > ending wavelength

    :ivar bool _custom_wave: True if Instrument's wavelength has been manually set
    :ivar np.ndarray wave: Instrument's wavelength grid
    :ivar Optional[Union[float,np.ndarray]] snr: Signal/Noise of observation (per pixel). Initially None.
    """

    def __init__(
        self,
        name: str,
        res: float,
        samp: float,
        start: float,
        end: float,
        truncate: bool = False,
    ) -> None:
        self.name = name
        self.R_res = res
        self.R_samp = samp
        self.start_wavelength = start
        self.end_wavelength = end
        self.wave = generate_wavelength_template(
            self.start_wavelength,
            self.end_wavelength,
            self.R_res,
            self.R_samp,
            truncate=truncate,
        )
        self._custom_wave = False
        self.snr = None

    def set_custom_wave(self, wave: np.ndarray, update_config: bool = True) -> None:
        """
        Sets instrument wavelength array to input array

        :param np.ndarray wave: Array of wavelengths
        :param bool update_config: Update instrument's start_wavelength and end_wavelength attributes to match new wavelength grid.
        :return:
        """
        self.wave = wave
        if update_config:
            self.start_wavelength = self.wave[0]
            self.end_wavelength = self.wave[-1]
            self.R_samp = np.nan
        self._custom_wave = True

    def reset_wave(self, truncate: bool = False) -> None:
        """
        Reset wavelength based on instrument configuration
        (beginning and ending wavelengths, resolving power, and sampling of the resolution element).

        :param bool truncate: truncate any pixels with lambda > ending wavelength.
        :return:
        """
        if not np.isfinite(self.R_samp):
            raise ValueError(f"R_samp must be an int or a float, not {self.R_samp}")
        self.wave = generate_wavelength_template(
            self.start_wavelength,
            self.end_wavelength,
            self.R_res,
            self.R_samp,
            truncate=truncate,
        )
        self._custom_wave = False

    def set_snr(
        self, snr_input: Union[int, float, np.ndarray, Sig2NoiseQuery], fill_value=None
    ) -> None:
        """
        Sets S/N for instrument configuration.

        - If snr_input is an int or float, a constant S/N is set for all pixels.
        - If snr_input is a 2D array, the first row is the wavelength grid and the second row is the S/N per pixel. The S/N is then interpolated onto the instrument's wavelength grid.
        - If snr_input is a 1D array, the wavelength grid is assumed to be linearly spaced from the instruments starting and ending wavelength. The S/N is then interpolated onto the instrument's wavelength grid.
        - If snr_input is a Sig2NoiseQuery, the relevant ETC is queried and the S/N is interpolated onto the instrument's wavelength grid.

        :param snr_input: Signal-to-Noise Ratio.
        :param fill_value: Argument passed on to scipy.interpolate.interp1d to handle S/N for wavelength regions that extend
            beyond the coverage of snr_input. Common choices are "extrapolate" (to linearly extrapolate) or 0
            (to set S/N to zero).
            If None, an error will be raised if snr_input does not span the full wavelength range of the instrument.
        :return:
        """
        if (
            (type(snr_input) == int)
            or (type(snr_input) == float)
            or (type(snr_input) == np.float64)
        ):
            self.snr = snr_input * np.ones_like(self.wave)
        elif isinstance(snr_input, np.ndarray):
            if snr_input.ndim == 2:
                if fill_value is not None:
                    snr_interpolator = interp1d(
                        snr_input[0],
                        snr_input[1],
                        bounds_error=False,
                        fill_value=fill_value,
                    )
                else:
                    snr_interpolator = interp1d(
                        snr_input[0], snr_input[1], bounds_error=True,
                    )
                self.snr = snr_interpolator(self.wave)
            elif snr_input.ndim == 1:
                warn(
                    f"snr_input is a 1D array. Assuming a linearly spaced wavelength grid from {self.wave.min()} to {self.wave.max()} Angstrom",
                    UserWarning,
                )
                fake_wave = np.linspace(
                    self.wave.min(), self.wave.max(), snr_input.shape[0]
                )
                snr_interpolator = interp1d(
                    fake_wave, snr_input, bounds_error=False, fill_value="extrapolate"
                )
                self.snr = snr_interpolator(self.wave)
            else:
                raise ValueError("S/N array must have ndim <= 2")
        elif isinstance(snr_input, (Sig2NoiseQuery)):
            snr_return = snr_input.query_s2n()
            if fill_value is not None:
                snr_interpolator = interp1d(
                    snr_return[0],
                    snr_return[1],
                    bounds_error=False,
                    fill_value=fill_value,
                )
            else:
                snr_interpolator = interp1d(
                    snr_return[0], snr_return[1], bounds_error=True,
                )
            self.snr = snr_interpolator(self.wave)
        else:
            raise ValueError("Cannot parse snr_input")
        self.snr[self.snr < 0] = 0

    def summary(self) -> None:
        """
        Prints summary of instrument configuration.

        :return:
        """
        if not self._custom_wave:
            print(
                f"{self.name}\n"
                + f"{self.start_wavelength} < lambda (A) < {self.end_wavelength}\n"
                + f"R = {self.R_res}\n"
                + f"Sampling = {self.R_samp} pix/FWHM"
            )
        else:
            print(
                f"{self.name}\n"
                + f"Custom Wavelength w/\n"
                + f"({self.wave[0]} < lambda (A) < {self.wave[-1]})\n"
                + f"R = {self.R_res}\n"
            )

    def __len__(self) -> int:
        """
        :return int: length of wavelength array
        """
        return len(self.wave)


class AllInstruments:
    """
    Object containing all pre-defined instrument configurations.

    :param file: JSON file containing  instrument parameters. If None, uses default Chem-I-Calc instrument file.

    :ivar Dict[str,InstConfig] spectrographs: Dictionary of instrument configurations
    """

    def __init__(self, file: str = None) -> None:
        if file == None:
            file = inst_file
        with open(file) as f:
            all_inst_dict = json.load(f)
        self.spectrographs = {}
        for observatory, obs_dict in all_inst_dict.items():
            for instrument, inst_dict in obs_dict.items():
                if inst_dict["samp"] == 0:
                    inst_dict["samp"] = sampX
                self.spectrographs[instrument] = InstConfig(
                    name=inst_dict["name"],
                    res=inst_dict["res"],
                    samp=inst_dict["samp"],
                    start=inst_dict["start"],
                    end=inst_dict["end"],
                    truncate=False,
                )

    def list_spectrographs(self) -> None:
        """
        Lists names of all predefined instruments

        :return:
        """
        for key, item in self.spectrographs.items():
            item.summary()
            print("\n")

    def get_spectrograph(self, name: str) -> InstConfig:
        """
        Get InstConfig object of a predefined instrument

        :param str name: name of spectrograph
        :return: Predefined InstConfig
        """
        return copy.deepcopy(self.spectrographs[name])


AllInst = AllInstruments()
"""
AllInstruments: Pre-initialized object of all the instruments
"""
