from typing import Union
import copy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from chemicalc.utils import (
    generate_wavelength_template,
)
from chemicalc.s2n import Sig2NoiseWMKO


sampX: float = 3  # Placeholder for Unknown Wavelength Sampling


class InstConfig:
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
        self.custom_wave = False
        self.snr = 100 * np.ones_like(self.wave)

    def set_custom_wave(self, wave: np.ndarray, update_config: bool = True) -> None:
        """

        :param np.ndarray wave: Array of wavelengths
        :param bool update_config: update instrument's beginning and ending wavelength config
        :return:
        """
        self.wave = wave
        if update_config:
            self.start_wavelength = self.wave[0]
            self.end_wavelength = self.wave[-1]
            self.R_samp = np.nan
        self.custom_wave = True

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
        self.custom_wave = False

    def set_snr(self, snr_input: Union[int, float, np.ndarray, Sig2NoiseWMKO]) -> None:
        """
        Sets S/N for instrument
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
        elif isinstance(snr_input, Sig2NoiseWMKO):
            self.snr = snr_input.query_s2n(wavelength=self.wave)
        else:
            raise ValueError("Cannot parse snr_input")
        self.snr[self.snr < 0] = 0

    def summary(self) -> None:
        """
        Prints summary of instrument configuration. Does not update if
        :return:
        """
        if self.custom_wave == True:
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
                + "({self.wave[0]} < lambda (A) < {self.self.wave[-1]})\n"
                + f"R ~ {self.R_res}\n"
            )

    def __len__(self) -> int:
        return len(self.wave)


class DEIMOS(InstConfig):
    def __init__(self, name, res, samp, start, end) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = "WMKO"
        self.instrument = "DEIMOS"
        self.mode = "Multi-Object Spectrograph"


class LRIS(InstConfig):
    def __init__(self, name, res, samp, start, end) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = "WMKO"
        self.instrument = "LRIS"
        self.mode = "Multi-Object Spectrograph"


class HIRES(InstConfig):
    def __init__(self, name, res, samp, start, end) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = "WMKO"
        self.instrument = "HIRES"
        self.mode = "Echelle Spectrograph"


class MIKE(InstConfig):
    def __init__(self, name, res, samp, start, end) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = "Magellan"
        self.instrument = "MIKE"
        self.mode = "Echelle Spectrograph"


class M2FS(InstConfig):
    def __init__(self, name, res, samp, start, end) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = "Magellan"
        self.instrument = "M2FS"
        self.mode = "Multi-Object Spectrograph"


class MiscInstrument(InstConfig):
    def __init__(
        self, name, res, samp, start, end, facility=None, instrument=None, mode=None
    ) -> None:
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = facility
        self.instrument = instrument
        self.mode = mode


class AllInstruments:
    def __init__(self) -> None:
        self.spectrographs = {  # # # WMKO # # #
            "DEIMOS 1200G": DEIMOS(
                "DEIMOS 1200G", res=6500, samp=4, start=6500, end=9000
            ),
            "DEIMOS 600ZD": DEIMOS(
                "DEIMOS 600ZD", res=2100, samp=5, start=4000, end=9100
            ),
            "DEIMOS 900ZD": DEIMOS(
                "DEIMOS 900ZD", res=2600, samp=5, start=4000, end=7500
            ),
            "DEIMOS 1200B": DEIMOS(
                "DEIMOS 1200B", res=4000, samp=4, start=4000, end=6600
            ),
            "LRIS 600/4000 (b)": LRIS(
                "LRIS 600/4000 (b)", res=1800, samp=4, start=3900, end=5500
            ),
            "LRIS 1200/7500 (r)": LRIS(
                "LRIS 1200/7500 (r)", res=4000, samp=5, start=7700, end=9000
            ),
            'HIRESr 1.0"': HIRES(
                'HIRESr 1.0"', res=35000, samp=3, start=3900, end=8350
            ),
            'HIRESr 0.8"': HIRES(
                'HIRESr 0.8"', res=49000, samp=3, start=3900, end=8350
            ),
            "KCWI Small BL": MiscInstrument(
                "KCWI Small BL", res=3600, samp=sampX, start=3500, end=5600
            ),  # dlambda = 2000
            "KCWI Medium BL": MiscInstrument(
                "KCWI Medium BL", res=1800, samp=sampX, start=3500, end=5600
            ),  # dlambda = 2000
            "KCWI Large BL": MiscInstrument(
                "KCWI Large BL", res=900, samp=sampX, start=3500, end=5600
            ),  # dlambda = 2000
            "KCWI Small BM": MiscInstrument(
                "KCWI Small BM", res=8000, samp=sampX, start=3500, end=5500
            ),  # dlambda = 800-900
            "KCWI Medium BM": MiscInstrument(
                "KCWI Medium BM", res=4000, samp=sampX, start=3500, end=5500
            ),  # dlambda = 800-900
            "KCWI Large BM": MiscInstrument(
                "KCWI Large BM", res=2000, samp=sampX, start=3500, end=5500
            ),  # dlambda = 800-900
            "KCWI Small BH1": MiscInstrument(
                "KCWI Small BH1", res=8000, samp=sampX, start=3500, end=4100
            ),  # dlambda = 400
            "KCWI Medium BH1": MiscInstrument(
                "KCWI Medium BH1", res=4000, samp=sampX, start=3500, end=4100
            ),  # dlambda = 400
            "KCWI Large BH1": MiscInstrument(
                "KCWI Large BH1", res=2000, samp=sampX, start=3500, end=4100
            ),  # dlambda = 400
            "KCWI Small BH2": MiscInstrument(
                "KCWI Small BH2", res=8000, samp=sampX, start=4000, end=4800
            ),  # dlambda = 370-440
            "KCWI Medium BH2": MiscInstrument(
                "KCWI Medium BH2", res=4000, samp=sampX, start=4000, end=4800
            ),  # dlambda = 370-440
            "KCWI Large BH2": MiscInstrument(
                "KCWI Large BH2", res=2000, samp=sampX, start=4000, end=4800
            ),  # dlambda = 370-440
            "KCWI Small BH3": MiscInstrument(
                "KCWI Small BH3", res=8000, samp=sampX, start=4700, end=5600
            ),  # dlambda = 470-530
            "KCWI Medium BH3": MiscInstrument(
                "KCWI Medium BH3", res=4000, samp=sampX, start=4700, end=5600
            ),  # dlambda = 470-530
            "KCWI Large BH3": MiscInstrument(
                "KCWI Large BH3", res=2000, samp=sampX, start=4700, end=5600
            ),  # dlambda = 470-530
            # # # Magellan # # #
            'MIKE 1" (r)': MIKE(
                'MIKE 1" (r)', res=22000, samp=3, start=5000, end=10000
            ),
            'MIKE 1" (b)': MIKE('MIKE 1" (b)', res=28000, samp=4, start=3500, end=5000),
            "M2FS MedRes": M2FS(
                "M2FS MedRes", res=10000, samp=sampX, start=5100, end=5315
            ),
            "M2FS HiRes": M2FS(
                "M2FS HiRes", res=18000, samp=sampX, start=5130, end=5185
            ),
            # # # MMT # # #
            "Hectochelle": MiscInstrument(
                "Hectochelle", res=20000, samp=6, start=5160, end=5280
            ),
            "Hectospec 270": MiscInstrument(
                "Hectospec 270", res=1500, samp=5, start=3900, end=9200
            ),
            "Hectospec 600": MiscInstrument(
                "Hectospec 600", res=5000, samp=5, start=5300, end=7800
            ),
            "Binospec 270": MiscInstrument(
                "Binospec 270", res=1300, samp=4, start=3900, end=9200
            ),
            "Binospec 600a": MiscInstrument(
                "Binospec 600a", res=2700, samp=3, start=4500, end=7000
            ),
            "Binospec 600b": MiscInstrument(
                "Binospec 600b", res=3600, samp=3, start=6000, end=8500
            ),
            "Binospec 600c": MiscInstrument(
                "Binospec 600c", res=4400, samp=3, start=7250, end=9750
            ),
            "Binospec 1000": MiscInstrument(
                "Binospec 1000", res=3900, samp=3, start=3900, end=5400
            ),
            # # # VLT # # #
            "MUSE": MiscInstrument("MUSE", res=2500, samp=sampX, start=4800, end=9300),
            "XSHOOTER (UVB)": MiscInstrument(
                "XSHOOTER (UVB)", res=6700, samp=5, start=3000, end=5500
            ),
            "XSHOOTER (VIS)": MiscInstrument(
                "XSHOOTER (VIS)", res=11400, samp=4, start=5500, end=10200
            ),
            "XSHOOTER (NIR)": MiscInstrument(
                "XSHOOTER (NIR)", res=5600, samp=4, start=10200, end=18000
            ),
            "GIRAFFE LR8": MiscInstrument(
                "GIRAFFE LR8", res=6500, samp=sampX, start=8200, end=9400
            ),
            "GIRAFFE HR10": MiscInstrument(
                "GIRAFFE HR10", res=19800, samp=sampX, start=5340, end=5620
            ),
            "GIRAFFE HR13": MiscInstrument(
                "GIRAFFE HR13", res=22500, samp=sampX, start=6120, end=6400
            ),
            "GIRAFFE HR14A": MiscInstrument(
                "GIRAFFE HR14A", res=28800, samp=sampX, start=6400, end=6620
            ),
            "GIRAFFE HR15": MiscInstrument(
                "GIRAFFE HR15", res=19300, samp=sampX, start=6620, end=6960
            ),
            # # # LBT # # #
            "MODS (b)": MiscInstrument(
                "MODS (b)", res=1850, samp=4, start=3200, end=5500
            ),
            "MODS (r)": MiscInstrument(
                "MODS (r)", res=2300, samp=4, start=5500, end=10500
            ),
            # # # JWST # # #
            "NIRSpec G140M/F070LP": MiscInstrument(
                "NIRSpec G140M/F070LP", res=1000, samp=sampX, start=7000, end=12700
            ),
            "NIRSpec G140M/F100LP": MiscInstrument(
                "NIRSpec G140M/F100LP", res=1000, samp=sampX, start=9700, end=17999
            ),
            "NIRSpec G140H/F070LP": MiscInstrument(
                "NIRSpec G140H/F070LP", res=2700, samp=sampX, start=7000, end=12700
            ),
            "NIRSpec G140H/F100LP": MiscInstrument(
                "NIRSpec G140H/F100LP", res=2700, samp=sampX, start=9700, end=17999
            ),
            # # # MSE # # #
            "MSE LR (b)": MiscInstrument(
                "MSE LR (b)", res=3000, samp=sampX, start=3600, end=5400
            ),
            "MSE LR (g)": MiscInstrument(
                "MSE LR (g)", res=3000, samp=sampX, start=5400, end=7200
            ),
            "MSE LR (r)": MiscInstrument(
                "MSE LR (r)", res=3000, samp=sampX, start=7200, end=9500
            ),
            "MSE LR (NIR)": MiscInstrument(
                "MSE LR (NIR)", res=3000, samp=sampX, start=9500, end=13000
            ),
            "MSE MR (b)": MiscInstrument(
                "MSE MR (b)", res=5000, samp=sampX, start=3900, end=5000
            ),
            "MSE MR (g)": MiscInstrument(
                "MSE MR (g)", res=5000, samp=sampX, start=5750, end=6900
            ),
            "MSE MR (r)": MiscInstrument(
                "MSE MR (r)", res=5000, samp=sampX, start=7370, end=9000
            ),
            # # # MW Surveys # # #
            "LAMOST": MiscInstrument(
                "LAMOST", res=2000, samp=sampX, start=3900, end=9000
            ),
            "WEAVE": MiscInstrument(
                "WEAVE", res=6000, samp=sampX, start=3700, end=10000
            ),
            "RAVE": MiscInstrument("RAVE", res=8000, samp=sampX, start=8400, end=8800),
            "DESI (b)": MiscInstrument(
                "DESI (b)", res=2500, samp=3, start=3600, end=5550
            ),
            "DESI (r)": MiscInstrument(
                "DESI (r)", res=3500, samp=3, start=5550, end=6560
            ),
            "DESI (i)": MiscInstrument(
                "DESI (i)", res=4500, samp=3, start=6560, end=9800
            ),
            "R100": MiscInstrument("R100", res=1e2, samp=sampX, start=3001, end=17999),
            "R1000": MiscInstrument(
                "R1000", res=1e3, samp=sampX, start=3001, end=17999
            ),
            "R10000": MiscInstrument(
                "R10000", res=1e4, samp=sampX, start=3001, end=17999
            ),
        }

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
