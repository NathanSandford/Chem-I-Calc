from typing import Union
import copy
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import interp1d
from chemicalc.utils import generate_wavelength_template
from chemicalc.s2n import Sig2NoiseWMKO, Sig2NoiseVLT, Sig2NoiseMSE
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

    Attributes:
        _custom_wave (bool): True if Instrument's wavelength has been manually set
        wave (np.ndarray): Instrument's wavelength grid
        snr (Optional[Union[float, np.ndarray]]): Signal/Noise of observation. Initially None.
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
        :param bool update_config: Update instrument's start_wavelength and end_wavelength attributes
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

    def set_snr(self, snr_input: Union[int, float, np.ndarray, Sig2NoiseWMKO]) -> None:
        """
        Sets S/N for instrument configuration

        :param snr_input: Signal-to-Noise Ratio
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
                snr_interpolator = interp1d(
                    snr_input[0],
                    snr_input[1],
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                self.snr = snr_interpolator(self.wave)
            elif snr_input.ndim == 1:
                fake_wave = np.linspace(
                    self.wave.min(), self.wave.max(), snr_input.shape[0]
                )
                snr_interpolator = interp1d(
                    fake_wave, snr_input, bounds_error=False, fill_value="extrapolate"
                )
                self.snr = snr_interpolator(self.wave)
            else:
                raise ValueError("S/N array must have ndim <= 2")
        elif isinstance(snr_input, (Sig2NoiseWMKO, Sig2NoiseVLT,  Sig2NoiseMSE)):
            self.snr = snr_input.query_s2n(wavelength=self.wave)
        else:
            raise ValueError("Cannot parse snr_input")
        self.snr[self.snr < 0] = 0

    def summary(self) -> None:
        """
        Prints summary of instrument configuration.

        :return:
        """
        if self._custom_wave == False:
            print(
                f"{self.name}\n"
                + f"{self.start_wavelength} < lambda (A) < {self.end_wavelength}\n"
                + f"R ~ {self.R_res}\n"
                + f"Sampling ~ {self.R_samp} pix/FWHM"
            )
        else:
            print(
                f"{self.name}\n"
                + f"Custom Wavelength w/\n"
                + f"({self.wave[0]} < lambda (A) < {self.wave[-1]})\n"
                + f"R ~ {self.R_res}\n"
            )

    def __len__(self) -> int:
        """
        :return int: length of wavelength array
        """
        return len(self.wave)


class AllInstruments:
    """
    Object containing all pre-defined instrument configurations.

    Attributes:
        spectrographs (dict[str, InstConfig]): Dictionary of instrument configurations

    """
    def __init__(self) -> None:
        with open(inst_file) as f:
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
        print(list(self.spectrographs.keys()))

    def get_spectrograph(self, name: str) -> InstConfig:
        """
        Get InstConfig object of a predefined instrument

        :param str name: name of spectrograph
        :return: Predefined InstConfig
        """
        return copy.deepcopy(self.spectrographs[name])
