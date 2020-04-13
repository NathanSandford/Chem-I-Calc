from typing import Optional, Union, Tuple
import os
import time
import numpy as np
from scipy.interpolate import interp1d
import mechanicalsoup
import requests
import json
from chemicalc.utils import decode_base64_dict
from chemicalc.file_mgmt import data_dir
from enyo.etc import spectrum, efficiency, telescopes, aperture, detector, extract
from enyo.etc.observe import Observation
from enyo.scripts.fobos_etc import (
    get_wavelength_vector,
    read_emission_line_database,
    get_spectrum,
    get_source_distribution,
)


keck_options = {
    "instrument": ["lris", "deimos", "hires"],
    "mag type": ["Vega", "AB"],
    "filter": [
        "sdss_r.dat",
        "sdss_g.dat",
        "sdss_i.dat",
        "sdss_u.dat",
        "sdss_z.dat",
        "Buser_B.dat",
        "Buser_V.dat",
        "Cousins_R.dat",
        "Cousins_I.dat",
    ],
    "template": [
        "O5V_pickles_1.fits",
        "B5V_pickles_6.fits",
        "A0V_pickles_9.fits",
        "A5V_pickles_12.fits",
        "F5V_pickles_16.fits",
        "G5V_pickles_27.fits",
        "K0V_pickles_32.fits",
        "K5V_pickles_36.fits",
        "M5V_pickles_44.fits",
    ],
    "grating (DEIMOS)": ["600Z", "900Z", "1200G", "1200B"],
    "grating (LRIS)": ["600/7500", "600/10000", "1200/9000", "400/8500", "831/8200"],
    "grism (LRIS)": ["B300", "B600"],
    "binning (DEIMOS)": ["1x1"],
    "binning (LRIS)": ["1x1", "2x1", "2x2", "3x1"],
    "binning (ESI)": ["1x1", "2x2", "2x1", "3x1"],
    "binning (HIRES)": ["1x1", "2x1", "2x2", "3x1"],
    "slitwidth (DEIMOS)": ["0.75", "1.0", "1.5"],
    "slitwidth (LRIS)": ["0.7", "1.0", "1.5"],
    "slitwidth (ESI)": ["0.75", "0.3", "0.5", "1.0"],
    "slitwidth (HIRES)": ["C5", "E4", "B2", "B5", "E5", "D3"],
    "slitwidth arcsec (HIRES)": [1.15, 0.40, 0.57, 0.86, 0.80, 1.72],
    "dichroic (LRIS)": ["D560"],
    "central wavelength (DEIMOS)": ["5000", "6000", "7000", "8000"],
}
mse_options = {
    "spec_mode": ["LR", "MR", "HR"],
    "airmass": ["1.0", "1.2", "1.5"],
    "filter": ["u", "g", "r", "i", "z", "Y", "J"],
    "src_type": ["extended", "point"],
    "template": [
        "o5v",
        "o9v",
        "b1v",
        "b2ic",
        "b3v",
        "b8v",
        "b9iii",
        "b9v",
        "a0iii",
        "a0v",
        "a2v",
        "f0v",
        "g0i",
        "g2v",
        "g5iii",
        "k2v",
        "k7v",
        "m2v",
        "flat",
        "WD",
        "LBG_EW_le_0",
        "LBG_EW_0_20",
        "LBG_EW_ge_20",
        "qso1",
        "qso2",
        "elliptical",
        "spiral_Sc",
        "HII",
        "PN",
    ],
}
etc_file_dir = data_dir.joinpath("etc_files")


class Sig2NoiseWMKO:
    def __init__(
        self,
        instrument,
        exptime,
        mag,
        template,
        magtype="Vega",
        band="Cousins_I.dat",
        airmass=1.1,
        seeing=0.75,
        redshift=0,
    ):
        if instrument not in keck_options["instrument"]:
            raise KeyError(f"{instrument} not one of {keck_options['instrument']}")
        if magtype not in keck_options["mag type"]:
            raise KeyError(f"{magtype} not one of {keck_options['mag type']}")
        if band not in keck_options["filter"]:
            raise KeyError(f"{band} not one of {keck_options['filter']}")
        if template not in keck_options["template"]:
            raise KeyError(f"{template} not one of {keck_options['template']}")
        self.instrument = instrument
        self.mag = mag
        self.magtype = magtype
        self.filter = band
        self.template = template
        self.exptime = exptime
        self.airmass = airmass
        self.seeing = seeing
        self.redshift = redshift

    def query_s2n(self, wavelength="default"):
        raise NotImplementedError(
            "No generic S/N query, see specific instrument children classes"
        )


class Sig2NoiseDEIMOS(Sig2NoiseWMKO):
    def __init__(
        self,
        grating,
        exptime,
        mag,
        template,
        magtype="Vega",
        band="Cousins_I.dat",
        cwave="7000",
        slitwidth="0.75",
        binning="1x1",
        airmass=1.1,
        seeing=0.75,
        redshift=0,
    ):
        Sig2NoiseWMKO.__init__(
            self,
            "deimos",
            exptime,
            mag,
            template,
            magtype,
            band,
            airmass,
            seeing,
            redshift,
        )
        if grating not in keck_options["grating (DEIMOS)"]:
            raise KeyError(f"{grating} not one of {keck_options['grating (DEIMOS)']}")
        if binning not in keck_options["binning (DEIMOS)"]:
            raise KeyError(f"{binning} not one of {keck_options['binning (DEIMOS)']}")
        if slitwidth not in keck_options["slitwidth (DEIMOS)"]:
            raise KeyError(
                f"{slitwidth} not one of {keck_options['slitwidth (DEIMOS)']}"
            )
        if cwave not in keck_options["central wavelength (DEIMOS)"]:
            raise KeyError(
                f"{cwave} not one of {keck_options['central wavelength (DEIMOS)']}"
            )
        self.grating = grating
        self.binning = binning
        self.slitwidth = slitwidth
        self.cwave = cwave

    def query_s2n(self, wavelength="default"):
        url = "http://etc.ucolick.org/web_s2n/deimos"
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form["grating"] = self.grating
        form["cwave"] = self.cwave
        form["slitwidth"] = self.slitwidth
        form["binning"] = self.binning
        form["exptime"] = str(self.exptime)
        form["mag"] = str(self.mag)
        form["ffilter"] = self.filter
        if self.magtype.lower() == "vega":
            form["mtype"] = "1"
        elif self.magtype.lower() == "ab":
            form["mtype"] = "2"
        form["seeing"] = str(self.seeing)
        form["template"] = self.template
        form["airmass"] = str(self.airmass)
        form["redshift"] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data["s2n"]).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == "default":
            return snr
        else:
            raise ValueError("Wavelength input not recognized")


class Sig2NoiseLRIS(Sig2NoiseWMKO):
    def __init__(
        self,
        grating,
        grism,
        exptime,
        mag,
        template,
        magtype="Vega",
        band="Cousins_I.dat",
        dichroic="D560",
        slitwidth="0.7",
        binning="1x1",
        airmass=1.1,
        seeing=0.75,
        redshift=0,
    ):
        Sig2NoiseWMKO.__init__(
            self,
            "lris",
            exptime,
            mag,
            template,
            magtype,
            band,
            airmass,
            seeing,
            redshift,
        )
        if grating not in keck_options["grating (LRIS)"]:
            raise KeyError(f"{grating} not one of {keck_options['grating (LRIS)']}")
        if grism not in keck_options["grism (LRIS)"]:
            raise KeyError(f"{grism} not one of {keck_options['grism (LRIS)']}")
        if binning not in keck_options["binning (LRIS)"]:
            raise KeyError(f"{binning} not one of {keck_options['binning (LRIS)']}")
        if slitwidth not in keck_options["slitwidth (LRIS)"]:
            raise KeyError(f"{slitwidth} not one of {keck_options['slitwidth (LRIS)']}")
        if dichroic not in keck_options["dichroic (LRIS)"]:
            raise KeyError(f"{dichroic} not one of {keck_options['dichroic (LRIS)']}")
        self.grating = grating
        self.grism = grism
        self.binning = binning
        self.slitwidth = slitwidth
        self.dichroic = dichroic

    def query_s2n(self, wavelength="default"):
        url = "http://etc.ucolick.org/web_s2n/lris"
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form["grating"] = self.grating
        form["grism"] = self.grism
        form["dichroic"] = self.dichroic
        form["slitwidth"] = self.slitwidth
        form["binning"] = self.binning
        form["exptime"] = str(self.exptime)
        form["mag"] = str(self.mag)
        form["ffilter"] = self.filter
        if self.magtype.lower() == "vega":
            form["mtype"] = "1"
        elif self.magtype.lower() == "ab":
            form["mtype"] = "2"
        form["seeing"] = str(self.seeing)
        form["template"] = self.template
        form["airmass"] = str(self.airmass)
        form["redshift"] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data["s2n"]).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == "default":
            return snr
        else:
            raise ValueError("Wavelength input not recognized")


class Sig2NoiseESI(Sig2NoiseWMKO):
    def __init__(
        self,
        exptime,
        mag,
        template,
        magtype="Vega",
        band="Cousins_I.dat",
        dichroic="D560",
        slitwidth="0.75",
        binning="1x1",
        airmass=1.1,
        seeing=0.75,
        redshift=0,
    ):
        Sig2NoiseWMKO.__init__(
            self,
            "lris",
            exptime,
            mag,
            template,
            magtype,
            band,
            airmass,
            seeing,
            redshift,
        )
        if binning not in keck_options["binning (ESI)"]:
            raise KeyError(f"{binning} not one of {keck_options['binning (ESI)']}")
        if slitwidth not in keck_options["slitwidth (ESI)"]:
            raise KeyError(f"{slitwidth} not one of {keck_options['slitwidth (ESI)']}")
        self.binning = binning
        self.slitwidth = slitwidth

    def query_s2n(self, wavelength="default"):
        url = "http://etc.ucolick.org/web_s2n/esi"
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form["slitwidth"] = self.slitwidth
        form["binning"] = self.binning
        form["exptime"] = str(self.exptime)
        form["mag"] = str(self.mag)
        form["ffilter"] = self.filter
        if self.magtype.lower() == "vega":
            form["mtype"] = "1"
        elif self.magtype.lower() == "ab":
            form["mtype"] = "2"
        form["seeing"] = str(self.seeing)
        form["template"] = self.template
        form["airmass"] = str(self.airmass)
        form["redshift"] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data["s2n"]).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == "default":
            return snr
        else:
            raise ValueError("Wavelength input not recognized")


class Sig2NoiseHIRES(Sig2NoiseWMKO):
    def __init__(
        self,
        slitwidth,
        exptime,
        mag,
        template,
        magtype="Vega",
        band="Cousins_I.dat",
        binning="1x1",
        airmass=1.1,
        seeing=0.75,
        redshift=0,
    ):
        Sig2NoiseWMKO.__init__(
            self,
            "hires",
            exptime,
            mag,
            template,
            magtype,
            band,
            airmass,
            seeing,
            redshift,
        )
        if binning not in keck_options["binning (HIRES)"]:
            raise KeyError(f"{binning} not one of {keck_options['binning (HIRES)']}")
        if slitwidth not in keck_options["slitwidth (HIRES)"]:
            raise KeyError(
                f"{slitwidth} not one of {keck_options['slitwidth (HIRES)']}"
            )
        self.binning = binning
        self.slitwidth = slitwidth

    def query_s2n(self, wavelength="default"):
        url = "http://etc.ucolick.org/web_s2n/hires"
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form["slitwidth"] = self.slitwidth
        form["binning"] = self.binning
        form["exptime"] = str(self.exptime)
        form["mag"] = str(self.mag)
        form["ffilter"] = self.filter
        if self.magtype.lower() == "vega":
            form["mtype"] = "1"
        elif self.magtype.lower() == "ab":
            form["mtype"] = "2"
        form["seeing"] = str(self.seeing)
        form["template"] = self.template
        form["airmass"] = str(self.airmass)
        form["redshift"] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data["s2n"]).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == "default":
            return snr
        else:
            raise ValueError("Wavelength input not recognized")


def calculate_mods_snr(F, wave, t_exp, airmass=1.1, mode="dichroic", side=None):
    assert mode in ["dichroic", "direct"], 'Mode must be either "dichroic" or "direct"'
    if mode is "direct":
        assert side in ["red", "blue"], 'Side must be "red" or "blue" if mode is direct'
    assert len(F) == len(wave), "Flux and Wavelength must be the same length"
    log_t_exp = np.log10(t_exp)
    log_F = np.log10(F)
    g_blue = 2.5  # electron / ADU
    g_red = 2.6  # electron / ADU
    sigma_RO_red = 2.5  # electron
    sigma_RO_blue = 2.5  # electron
    A_per_pix_red = 0.85
    A_per_pix_blue = 0.50
    slit_loss_factor = 0.76
    atm_extinct_curve = np.genfromtxt(etc_file_dir.joinpath("LBTO_atm_extinct.txt")).T
    atm_extinct = interp1d(
        x=atm_extinct_curve[0],
        y=atm_extinct_curve[1],
        bounds_error=False,
        fill_value="extrapolate",
    )
    log_S_0_red = np.genfromtxt(etc_file_dir.joinpath("MODS_red_S_0.txt")).T
    log_S_0_blue = np.genfromtxt(etc_file_dir.joinpath("MODS_blue_S_0.txt")).T
    g = np.zeros_like(wave)
    if mode == "dichroic":
        log_S_0_r = interp1d(
            log_S_0_red[0], log_S_0_red[2], bounds_error=False, fill_value="extrapolate"
        )
        log_S_0_b = interp1d(
            log_S_0_blue[0],
            log_S_0_blue[2],
            bounds_error=False,
            fill_value="extrapolate",
        )
        log_S_red = (
            log_S_0_r(wave) + log_F + log_t_exp - 0.4 * atm_extinct(wave) * airmass
        )
        log_S_blue = (
            log_S_0_b(wave) + log_F + log_t_exp - 0.4 * atm_extinct(wave) * airmass
        )
        S_red = 10 ** log_S_red * slit_loss_factor
        S_blue = 10 ** log_S_blue * slit_loss_factor
        snr_red = (
            g_red * S_red / np.sqrt(g_red * S_red + sigma_RO_red ** 2) * A_per_pix_red
        )
        snr_blue = (
            g_blue
            * S_blue
            / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
            * A_per_pix_blue
        )
        snr = np.max([snr_red, snr_blue], axis=0)
    elif mode == "direct":
        if side == "red":
            log_S_0_r = interp1d(
                log_S_0_red[0],
                log_S_0_red[1],
                bounds_error=False,
                fill_value="extrapolate",
            )
            log_S_red = (
                log_S_0_r(wave) + log_F + log_t_exp - 0.4 * atm_extinct(wave) * airmass
            )
            S_red = 10 ** log_S_red * slit_loss_factor
            snr = (
                g_red
                * S_red
                / np.sqrt(g_red * S_red + sigma_RO_red ** 2)
                * A_per_pix_red
            )
        elif side == "blue":
            log_S_0_b = interp1d(
                log_S_0_blue[0],
                log_S_0_blue[1],
                bounds_error=False,
                fill_value="extrapolate",
            )
            log_S_blue = (
                log_S_0_b(wave) + log_F + log_t_exp - 0.4 * atm_extinct(wave) * airmass
            )
            S_blue = 10 ** log_S_blue * slit_loss_factor
            snr = (
                g_blue
                * S_blue
                / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
                * A_per_pix_blue
            )
    return np.array([wave, snr])


class Sig2NoiseMSE:
    def __init__(
        self,
        exptime,
        mag,
        template,
        spec_mode="LR",
        filter="g",
        airmass="1.2",
        seeing=0.5,
        skymag=20.7,
        src_type="point",
        redshift=0,
    ):
        self.url_base = "http://etc-dev.cfht.hawaii.edu/cgi-bin/mse/mse_wrapper.py"
        # Hard Coded Values
        self.sessionID = 1234
        self.coating = "ZeCoat"
        self.fibdiam = 1
        self.spatbin = 2
        self.specbin = 1
        self.meth = "getSNR"
        self.snr_value = 10
        # Check Values
        if template not in mse_options["template"]:
            raise KeyError(f"{template} not one of {mse_options['template']}")
        if spec_mode not in mse_options["spec_mode"]:
            raise KeyError(f"{spec_mode} not one of {mse_options['spec_mode']}")
        if filter not in mse_options["filter"]:
            raise KeyError(f"{filter} not one of {mse_options['filter']}")
        if airmass not in mse_options["airmass"]:
            raise KeyError(f"{airmass} not one of {mse_options['airmass']}")
        if src_type not in mse_options["src_type"]:
            raise KeyError(f"{src_type} not one of {mse_options['src_type']}")
        self.exptime = exptime
        self.mag = mag
        self.template = template
        self.spec_mode = spec_mode
        self.filter = filter
        self.airmass = airmass
        self.seeing = seeing
        self.skymag = skymag
        self.src_type = src_type
        self.redshift = redshift

    def query_s2n(
        self, wavelength="default", smoothed=False,
    ):
        url = (
            f"{self.url_base}?"
            + f"sessionID={self.sessionID}&"
            + f"coating={self.coating}&"
            + f"seeing={self.seeing}&"
            + f"airmass={self.airmass}&"
            + f"skymag={self.skymag}&"
            + f"spectro={self.spec_mode}&"
            + f"fibdiam={self.fibdiam}&"
            + f"spatbin={self.spatbin}&"
            + f"specbin={self.specbin}&"
            + f"meth={self.meth}&"
            + f"etime={self.exptime}&"
            + f"snr={self.snr_value}&"
            + f"src_type={self.src_type}&"
            + f"tgtmag={self.mag}&"
            + f"redshift={self.redshift}&"
            + f"band={self.filter}&"
            + f"template={self.template}"
        )
        response = requests.post(url)
        # Parse HTML response
        r = response.text.split("docs_json = '")[1].split("';")[0]
        model = json.loads(r)
        key = list(model.keys())[0]
        model_dict = model[key]
        model_pass1 = [
            _
            for _ in model_dict["roots"]["references"]
            if "data" in _["attributes"].keys()
        ]
        model_pass2 = [
            _ for _ in model_pass1 if "__ndarray__" in _["attributes"]["data"]["x"]
        ]
        x = {}
        y = {}
        for i, tmp in enumerate(model_pass2):
            x_str = tmp["attributes"]["data"]["x"]
            x[i] = decode_base64_dict(x_str)
            y_str = tmp["attributes"]["data"]["y"]
            y[i] = decode_base64_dict(y_str)
        # Sort Arrays
        order = np.argsort([array[0] for i, array in x.items()])
        x = {i: x[j] for i, j in enumerate(order)}
        y = {i: y[j] for i, j in enumerate(order)}
        x = {i: x[2 * i] for i in range(int(len(x) / 2))}
        if smoothed:
            y = {
                i: (
                    y[2 * i]
                    if (np.mean(y[2 * i]) > np.mean(y[2 * i + 1]))
                    else y[2 * i + 1]
                )
                for i in range(int(len(y) / 2))
            }
        else:
            y = {
                i: (
                    y[2 * i]
                    if (np.mean(y[2 * i]) < np.mean(y[2 * i + 1]))
                    else y[2 * i + 1]
                )
                for i in range(int(len(y) / 2))
            }
        y[0] = y[0][x[0] < x[1].min()]
        x[0] = x[0][x[0] < x[1].min()]
        y[1] = y[1][x[1] < x[2].min()]
        x[1] = x[1][x[1] < x[2].min()]
        y[2] = y[2][x[2] < x[3].min()]
        x[2] = x[2][x[2] < x[3].min()]
        filler_x = np.linspace(x[3].max(), x[4].min(), 100)
        filler_y = np.zeros(100)
        x = np.concatenate([x[0], x[1], x[2], x[3], filler_x, x[4]])
        y = np.concatenate([y[0], y[1], y[2], y[3], filler_y, y[4]])
        snr = np.vstack([x, y])
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == "default":
            return snr
        else:
            raise ValueError("Wavelength input not recognized")


def calculate_fobos_snr(
    spec_file: Optional[str] = None,
    spec_wave: Union[str, float] = "WAVE",
    spec_wave_units: str = "angstrom",
    spec_flux: Union[str, float] = "FLUX",
    spec_flux_units: Optional[str] = None,
    spec_res_indx: Optional[Union[str, float]] = None,
    spec_res_value: Optional[float] = None,
    spec_table: Optional[Union[str, float]] = None,
    mag: float = 24.0,
    mag_band: str = "g",
    mag_system: str = "AB",
    redshift: float = 0.0,
    emline: Optional[str] = None,
    sersic: Optional[Tuple[float, float, float, float]] = None,
    uniform: bool = False,
    exptime: float = 3600.0,
    fwhm: float = 0.65,
    airmass: float = 1.0,
    snr_units: str = "pixel",
    sky_err: float = 0.1,
    print_summary: bool = True,
):
    """
    This is slightly modified code from https://github.com/Keck-FOBOS/enyo/blob/master/python/enyo/scripts/fobos_etc.py
    :param spec_file:
    :param spec_wave:
    :param spec_wave_units:
    :param spec_flux:
    :param spec_flux_units:
    :param spec_res_indx:
    :param spec_res_value:
    :param spec_table:
    :param mag:
    :param mag_band:
    :param mag_system:
    :param redshift:
    :param emline:
    :param sersic:
    :param uniform:
    :param exptime:
    :param fwhm:
    :param airmass:
    :param snr_units:
    :param sky_err:
    :param print_summary:
    :return:
    """
    if sky_err < 0 or sky_err > 1:
        raise ValueError("--sky_err option must provide a value between 0 and 1.")
    # Constants:
    resolution = 3500.0  # lambda/dlambda
    fiber_diameter = 0.8  # Arcsec
    rn = 2.0  # Detector readnoise (e-)
    dark = 0.0  # Detector dark-current (e-/s)
    # Temporary numbers that assume a given spectrograph PSF and LSF.
    # Assume 3 pixels per spectral and spatial FWHM.
    spatial_fwhm = 3.0
    spectral_fwhm = 3.0
    # Get source spectrum in 1e-17 erg/s/cm^2/angstrom. Currently, the
    # source spectrum is assumed to be
    #   - normalized by the total integral of the source flux
    #   - independent of position within the source
    wavelengths = [3100, 10000, 4e-5]
    wave = get_wavelength_vector(wavelengths[0], wavelengths[1], wavelengths[2])
    emline_db = None if emline is None else read_emission_line_database(emline)
    spec = get_spectrum(
        wave,
        mag,
        mag_band=mag_band,
        mag_system=mag_system,
        spec_file=spec_file,
        spec_wave=spec_wave,
        spec_wave_units=spec_wave_units,
        spec_flux=spec_flux,
        spec_flux_units=spec_flux_units,
        spec_res_indx=spec_res_indx,
        spec_res_value=spec_res_value,
        spec_table=spec_table,
        emline_db=emline_db,
        redshift=redshift,
        resolution=resolution,
    )
    t = time.perf_counter()
    # Get the source distribution.  If the source is uniform, onsky is None.
    onsky = get_source_distribution(fwhm, uniform, sersic)
    # Get the sky spectrum
    sky_spectrum = spectrum.MaunakeaSkySpectrum()
    # Get the atmospheric throughput
    atmospheric_throughput = efficiency.AtmosphericThroughput(airmass=airmass)
    # Set the telescope. Defines the aperture area and throughput
    # (nominally 3 aluminum reflections for Keck)
    telescope = telescopes.KeckTelescope()
    # Define the observing aperture; fiber diameter is in arcseconds,
    # center is 0,0 to put the fiber on the target center. "resolution"
    # sets the resolution of the fiber rendering; it has nothing to do
    # with spatial or spectral resolution of the instrument
    fiber = aperture.FiberAperture(0, 0, fiber_diameter, resolution=100)
    # Get the spectrograph throughput (circa June 2018; TODO: needs to
    # be updated). Includes fibers + foreoptics + FRD + spectrograph +
    # detector QE (not sure about ADC). Because this is the total
    # throughput, define a generic efficiency object.
    thru_db = np.genfromtxt(
        os.path.join(os.environ["ENYO_DIR"], "data/efficiency", "fobos_throughput.db")
    )
    spectrograph_throughput = efficiency.Efficiency(thru_db[:, 1], wave=thru_db[:, 0])
    # System efficiency combines the spectrograph and the telescope
    system_throughput = efficiency.SystemThroughput(
        wave=spec.wave,
        spectrograph=spectrograph_throughput,
        telescope=telescope.throughput,
    )
    # Instantiate the detector; really just a container for the rn and
    # dark current for now. QE is included in fobos_throughput.db file,
    # so I set it to 1 here.
    det = detector.Detector(rn=rn, dark=dark, qe=1.0)
    # Extraction: makes simple assumptions about the detector PSF for
    # each fiber spectrum and mimics a "perfect" extraction, including
    # an assumption of no cross-talk between fibers. Ignore the
    # "spectral extraction".
    extraction = extract.Extraction(
        det,
        spatial_fwhm=spatial_fwhm,
        spatial_width=1.5 * spatial_fwhm,
        spectral_fwhm=spectral_fwhm,
        spectral_width=spectral_fwhm,
    )
    # Perform the observation
    obs = Observation(
        telescope,
        sky_spectrum,
        fiber,
        exptime,
        det,
        system_throughput=system_throughput,
        atmospheric_throughput=atmospheric_throughput,
        airmass=airmass,
        onsky_source_distribution=onsky,
        source_spectrum=spec,
        extraction=extraction,
        snr_units=snr_units,
    )
    # Construct the S/N spectrum
    snr = obs.snr(sky_sub=True, sky_err=sky_err)
    snr_label = 'S/N per {0}'.format('R element' if snr_units == 'resolution'
                                         else snr_units)
    if print_summary:
        # Report
        g = efficiency.FilterResponse(band='g')
        r = efficiency.FilterResponse(band='r')
        iband = efficiency.FilterResponse(band='i')
        print('-'*70)
        print('{0:^70}'.format('FOBOS S/N Calculation (v0.2)'))
        print('-'*70)
        print('Compute time: {0} seconds'.format(time.perf_counter() - t))
        print('Object g- and r-band AB magnitude: {0:.1f} {1:.1f}'.format(
                        spec.magnitude(band=g), spec.magnitude(band=r)))
        print('Sky g- and r-band AB surface brightness: {0:.1f} {1:.1f}'.format(
                        sky_spectrum.magnitude(band=g), sky_spectrum.magnitude(band=r)))
        print('Exposure time: {0:.1f} (s)'.format(exptime))
        if not uniform:
            print('Aperture Loss: {0:.1f}%'.format((1-obs.aperture_factor)*100))
        print('Extraction Loss: {0:.1f}%'.format((1-obs.extraction.spatial_efficiency)*100))
        print('Median {0}: {1:.1f}'.format(snr_label, np.median(snr.flux)))
        print('g-band weighted mean {0} {1:.1f}'.format(snr_label,
                    np.sum(g(snr.wave)*snr.flux)/np.sum(g(snr.wave))))
        print('r-band weighted mean {0} {1:.1f}'.format(snr_label,
                    np.sum(r(snr.wave)*snr.flux)/np.sum(r(snr.wave))))
        print('i-band weighted mean {0} {1:.1f}'.format(snr_label,
                    np.sum(iband(snr.wave)*snr.flux)/np.sum(iband(snr.wave))))
    return np.vstack([snr.wave, snr.flux])


def calculate_bluemuse_snr(wave, flux, exptime, nexp,
                           airmass=1.0, seeing=0.8, moon='d', pointsource=True,
                           nspatial=3, nspectral=1):
    """
    This code is adapted from https://git-cral.univ-lyon1.fr/johan.richard/BlueMUSE-ETC
    :param wave:
    :param flux:
    :param exptime:
    :param nexp:
    :param airmass:
    :param seeing:
    :param moon:
    :param pointsource:
    :param nspatial:
    :param nspectral:
    :return:
    """
    blueMUSE_etc_dir = etc_file_dir.joinpath('blueMUSE')
    ron = 3.0  # readout noise (e-)
    dcurrent = 3.0  # dark current (e-/pixel/s)
    spaxel = 0.3  # spaxel scale (arcsecs)
    fins = 0.2  # Instrument image quality (arcsecs)
    nbiases = 11  # number of biases used in calibration
    lmin = 3500.0  # minimum wavelength
    lmax = 6000.0  # maximum wavelength
    lstep = 0.66  # spectral sampling (Angstroms)
    lsf = lstep * 2.0  # in Angstroms
    musetrans = np.loadtxt(blueMUSE_etc_dir.joinpath('NewBlueMUSE_noatm.txt'))
    wmusetrans = musetrans[:, 0] * 10.0  # in Angstroms
    valmusetrans = musetrans[:, 1]
    tarea = 485000.0  # squared centimeters
    teldiam = 8.20  # diameter in meters
    h = 6.626196e-27  # erg.s

    wrange = np.arange(lmin, lmax, lstep)
    waveinput = wave
    fluxinput = flux
    flux = np.interp(wrange, waveinput, fluxinput)

    pixelarea = nspatial * nspatial * spaxel * spaxel  # in arcsec^2
    npixels = nspatial * nspatial * nspectral

    # Compute image quality as a function of seeing, airmass and wavelength
    iq = np.zeros(wrange.shape)
    frac = np.zeros(wrange.shape)
    snratio = np.zeros(wrange.shape)
    sky = np.zeros(wrange.shape)
    skyelectrons = np.zeros(wrange.shape)
    shape = np.array((101, 101))  # in 0.05 arcsec pixels
    yy, xx = np.mgrid[:shape[0], :shape[1]]

    def moffat(p, q):
        xdiff = p - 50.0
        ydiff = q - 50.0
        return (norm * (1 + (xdiff / a) ** 2 + (ydiff / a) ** 2) ** (-n))

    posmin = 50 - int(nspatial * spaxel / 2.0 / 0.05)  # in 0.05 arcsec pixels
    posmax = posmin + int(nspatial * spaxel / 0.05)  # in 0.05 arcsec pixels

    # For point sources compute the fraction of the total flux of the source
    if (pointsource):
        for k in range(wrange.shape[0]):
            # All of this is based on ESO atmosphere turbulence model (ETC)
            ftel = 0.0000212 * wrange[k] / teldiam  # Diffraction limit in arcsec
            r0 = 0.100 * (seeing ** -1) * (wrange[k] / 5000.0) ** 1.2 * (airmass) ** -0.6  # Fried parameter in meters
            fkolb = -0.981644
            if (r0 < 5.4):
                fatm = seeing * (airmass ** 0.6) * (wrange[k] / 5000.0) ** (-0.2) * np.sqrt(
                    (1. + fkolb * 2.183 * (r0 / 46.0) ** 0.356))
            else:
                fatm = 0.0
            # Full image quality FWHM in arcsecs
            iq[k] = np.sqrt(fatm * fatm + ftel * ftel + fins * fins)
            fwhm = iq[k] / 0.05  # FWHM of PSF in 0.05 arcsec pixels
            n = 2.5
            a = fwhm / (2 * np.sqrt(2 ** (1.0 / n) - 1.0))
            norm = (n - 1) / (np.pi * a * a)
            psf = moffat(yy, xx)
            # fraction of spatial PSF within extraction aperture
            frac[k] = np.sum(psf[posmin:posmax, posmin:posmax])

    # sky spectrum (grey moon)
    if (moon == 'g'):
        skyemtable = np.loadtxt(blueMUSE_etc_dir.joinpath('radiance_airmass1.0_0.5moon.txt'))
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = skyemtable[:, 1] * airmass  # in photons / s / m2 / micron / arcsec2 approximated at given airmass
    else:  # dark conditions - no moon
        skyemtable = np.loadtxt(blueMUSE_etc_dir.joinpath('radiance_airmass1.0_newmoon.txt'))  # sky spectrum (grey) - 0.5 FLI
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = skyemtable[:, 1] * airmass  # in photons / s / m2 / micron / arcsec2
    # Interpolate sky spectrum at instrumental wavelengths
    sky = np.interp(wrange, skyemw, skyemflux)
    # loads sky transmission
    atmtrans = np.loadtxt(blueMUSE_etc_dir.joinpath('transmission_airmass1.txt'))
    atmtransw = atmtrans[:, 0] * 10.0  # In Angstroms
    atmtransval = atmtrans[:, 1]
    atm = np.interp(wrange, atmtransw, atmtransval)
    # Interpolate transmission including sky transmission at corresponding airmass
    # Note: ESO ETC includes a 60% margin for MUSE
    transm = np.interp(wrange, wmusetrans, valmusetrans) * (atm ** (airmass))
    transmnoatm = np.interp(wrange, wmusetrans, valmusetrans)

    dit=exptime
    ndit=nexp
    for k in range(wrange.shape[0]):
        kmin = 1 + np.max([-1, int(k - nspectral / 2)])
        kmax = 1 + np.min([wrange.shape[0] - 1, int(k + nspectral / 2)])

        if (pointsource):
            signal = (np.sum(flux[kmin:kmax] * transm[kmin:kmax] * frac[kmin:kmax]) * lstep / (
                        h * 3e18 / wrange[k])) * tarea * dit * ndit  # in electrons
        else:  # extended source, flux is per arcsec2
            signal = (np.sum(flux[kmin:kmax] * transm[kmin:kmax]) * lstep / (
                        h * 3e18 / wrange[k])) * tarea * dit * ndit * pixelarea  # in electrons
        skysignal = (np.sum(sky[kmin:kmax] * transmnoatm[kmin:kmax]) * lstep / 10000.0) * (
                    tarea / 10000.0) * dit * ndit * pixelarea  # lstep converted in microns, tarea in m2                                  #in electrons
        skyelectrons[k] = skysignal
        noise = np.sqrt(ron * ron * npixels * (1.0 + 1.0 / nbiases) * ndit + dcurrent * (
                    dit * ndit / 3600.0) * npixels + signal + skysignal)  # in electrons
        snratio[k] = signal / noise
    return np.vstack([wrange, snratio])