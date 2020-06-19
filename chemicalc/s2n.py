from typing import Optional, Union, Tuple, List
from warnings import warn
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import mechanicalsoup
import requests
import json
from chemicalc.utils import decode_base64_dict, find_nearest_idx
from chemicalc.file_mgmt import etc_file_dir, download_bluemuse_files

wmko_options = {
    "instrument": ["lris", "deimos", "hires", "esi"],
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
mmt_options = {
    "inst_mode": [
        "BINOSPEC_1000",
        "BINOSPEC_270",
        "BINOSPEC_600",
        "HECTOSPEC_270",
        "HECTOSPEC_600",
    ],
    "template": [
        "O5V",
        "A0V",
        "A5V",
        "B0V",
        "F0V",
        "F5V",
        "G0V",
        "G2V",
        "K0V",
        "K5V",
        "M5V",
        "Moon",
    ],
    "filter": ["r_filt", "g_filt", "i_filt"],
    "aptype": ["Round", "Square", "Rectangular"],
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
vlt_options = {
    "instruments": ["UVES", "FLAMES-UVES", "FLAMES-GIRAFFE", "X-SHOOTER", "MUSE"],
    "src_target_mag_band (MUSE)": [
        "B",
        "V",
        "R",
        "I",
        "sloan_g_prime",
        "sloan_r_prime",
        "sloan_i_prime",
        "sloan_z_prime",
    ],
    "src_target_mag_band (GIRAFFE)": ["U", "B", "V", "R", "I",],
    "src_target_mag_band (UVES)": ["U", "B", "V", "R", "I",],
    "src_target_mag_band (X-SHOOTER)": ["U", "B", "V", "R", "I", "J", "H", "K",],
    "src_target_mag_system": ["Vega", "AB"],
    "src_target_type": ["template_spectrum"],
    "src_target_spec_type": [
        "Pickles_O5V",
        "Pickles_O9V",
        "Kurucz_B1V",
        "Pickles_B2IV",
        "Kurucz_B3V",
        "Kurucz_B8V",
        "Pickles_B9III",
        "Pickles_B9V",
        "Pickles_A0III",
        "Pickles_A0V",
        "Kurucz_A1V",
        "Kurucz_F0V",
        "Pickles_G0V",
        "Kurucz_G2V",
        "Pickles_K2V",
        "Pickles_K7V",
        "Pickles_M2V",
        "Planetary Nebula",
        "HII Region (ORION)",
        "Kinney_ell",
        "Kinney_s0",
        "Kinney_sa",
        "Kinney_sb",
        "Kinney_starb1",
        "Kinney_starb2",
        "Kinney_starb3",
        "Kinney_starb4",
        "Kinney_starb5",
        "Kinney_starb6",
        "Galev_E",
        "qso-interp",
    ],
    "sky_seeing": ["0.5", "0.6", "0.7", "0.8", "1.0", "1.3", "3.0"],
    "uves_det_cd_name": [
        "Blue_346",
        "Blue_437",
        "Red__520",
        "Red__580",
        "Red__600",
        "Red__860",
        "Dicroic1_Blue_346",
        "Dicroic2_Blue_346",
        "Dicroic1_Red__580",
        "Dicroic1_Blue_390",
        "Dicroic2_Blue_390",
        "Dicroic1_Red__564",
        "Dicroic2_Blue_437",
        "Dicroic2_red__760",
        "Dicroic2_Red__860",
    ],
    "uves_slit_width": [
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "1.0",
        "1.1",
        "1.2",
        "1.5",
        "1.8",
        "2.1",
        "2.4",
        "2.7",
        "3.0",
        "5.0",
        "10.0",
    ],
    "uves_ccd_binning": ["1x1", "1x1v", "2x2", "2x1", "3x2"],
    "giraffe_sky_sampling_mode": ["MEDUSA", "IFU052", "ARGUS052", "ARGUS030",],
    "giraffe_slicer": [
        "LR01",
        "LR02",
        "LR03",
        "LR04",
        "LR05",
        "LR06",
        "LR07",
        "LR08",
        "HR01",
        "HR02",
        "HR03",
        "HR04",
        "HR05A",
        "HR05B",
        "HR06",
        "HR07A",
        "HR07B",
        "HR08",
        "HR09A",
        "HR09B",
        "HR10",
        "HR11",
        "HR12",
        "HR13",
        "HR14A",
        "HR14B",
        "HR15",
        "HR15n",
        "HR16",
        "HR17A",
        "HR17B",
        "HR17B",
        "HR18",
        "HR19A",
        "HR19B",
        "HR20A",
        "HR20B",
        "HR21",
        "HR22A",
        "HR22B",
    ],
    "giraffe_ccd_mode": ["standard", "fast", "slow"],
    "xshooter_uvb_slitwidth": ["0.5", "0.8", "1.0", "1.3", "1.6", "5.0"],
    "xshooter_vis_slitwidth": ["0.4", "0.7", "0.9", "1.2", "1.5", "5.0"],
    "xshooter_nir_slitwidth": ["0.4", "0.6", "0.9", "1.2", "1.5", "5.0"],
    "xshooter_uvb_ccd_binning": [
        "high1x1slow",
        "high1x2slow",
        "high2x2slow",
        "low1x1fast",
        "low1x2fast",
        "low2x2fast",
    ],
    "xshooter_vis_ccd_binning": [
        "high1x1slow",
        "high1x2slow",
        "high2x2slow",
        "low1x1fast",
        "low1x2fast",
        "low2x2fast",
    ],
    "muse_mode": [
        "WFM_NONAO_N",  # Wide Field Mode without AO, nominal  wavelength range
        "WFM_NONAO_E",  # Wide Field Mode without AO, extended wavelength range
        "WFM_AO_N",  # Wide Field Mode with AO, nominal wavelength range
        "WFM_AO_E",  # Wide Field Mode with AO, extended wavelength range
        "NFM_AO_N",
    ],  # Narrow Field Mode with AO, nominal wavelength range
    "muse_spatial_binning": ["1", "2", "3", "4", "5", "10", "30", "60", "100"],
    "muse_spectra_binning": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "10",
        "20",
        "30",
        "40",
        "50",
        "100",
        "200",
        "400",
        "800",
        "1600",
        "3200",
    ],
}
lco_options = {
    "template": ["flat", "O5V", "B0V", "A0V", "F0V", "G0V", "K0V", "M0V"],
    "tempfilter": ["u", "g", "r", "i", "z"],
    "telescope": ["MAGELLAN1", "MAGELLAN2"],
    "MAGELLAN1_instrument": ["IMACS", "MAGE"],
    "MAGELLAN2_instrument": ["LDSS3", "MIKE"],
    "IMACS_mode": [
        "F2_150_11",
        "F2_200_15",
        "F2_300_17",
        "F2_300_26",
        "F4_150-3_3.4",
        "F4_300-4_6.0",
        "F4_600-8_9.3",
        "F4_600-13_14.0",
        "F4_1200-17_19.0",
        "F4_1200-27_27.0",
        "F4_1200-27_33.5",
    ],
    "MAGE_mode": ["ECHELLETTE"],
    "MIKE_mode": ["BLUE", "RED"],
    "LDSS3_mode": ["VPHALL", "VPHBLUE", "VPHRED"],
    "binspat": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "binspec": ["1", "2", "3", "4", "5", "6", "7", "8"],
    "nmoon": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
    ],
}


class Sig2NoiseQuery:
    """
    Base class for ETC queries
    """

    def __init__(self):
        pass

    def query_s2n(self) -> None:
        pass


class Sig2NoiseWMKO(Sig2NoiseQuery):
    """
    Superclass for WMKO ETC Queries

    :param str instrument: Keck instrument. Must be "DEIMOS", "LRIS", "HIRES", or "ESI"
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.wmko_options['template'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str band: Magnitude band. For valid options see s2n.wmko_options['filter'].
    :param float airmass: Airmass of observation
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float redshift: Redshift of the target
    """

    def __init__(
        self,
        instrument: str,
        exptime: float,
        mag: float,
        template: str,
        magtype: str = "Vega",
        band: str = "Cousins_I.dat",
        airmass: float = 1.1,
        seeing: float = 0.75,
        redshift: float = 0,
    ):
        Sig2NoiseQuery.__init__(self)
        if instrument not in wmko_options["instrument"]:
            raise KeyError(f"{instrument} not one of {wmko_options['instrument']}")
        if magtype not in wmko_options["mag type"]:
            raise KeyError(f"{magtype} not one of {wmko_options['mag type']}")
        if band not in wmko_options["filter"]:
            raise KeyError(f"{band} not one of {wmko_options['filter']}")
        if template not in wmko_options["template"]:
            raise KeyError(f"{template} not one of {wmko_options['template']}")
        self.instrument = instrument
        self.mag = mag
        self.magtype = magtype
        self.filter = band
        self.template = template
        self.exptime = exptime
        self.airmass = airmass
        self.seeing = seeing
        self.redshift = redshift

    def query_s2n(self) -> None:
        """
        No generic S/N query, see specific instrument subclasses

        :return:
        """
        raise NotImplementedError(
            "No generic S/N query, see specific instrument children classes"
        )


class Sig2NoiseDEIMOS(Sig2NoiseWMKO):
    """
    Keck/DEIMOS S/N Query (http://etc.ucolick.org/web_s2n/deimos)

    :param str grating: DEIMOS grating. Must be one of "600Z", "900Z", "1200G", or "1200B".
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.wmko_options['template'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str band: Magnitude band. For valid options see s2n.wmko_options['filter'].
    :param str cwave: Central wavelength of grating. Must be one of "5000", "6000", "7000", or "8000"
    :param str slitwidth: Width of slit in arcseconds. Must be "0.75", "1.0", or "1.5"
    :param str binning: spatial x spectral binning. "1x1" is the only option.
    :param flaot airmass: Airmass of observation
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float redshift: Redshift of the target
    """

    def __init__(
        self,
        grating: str,
        exptime: float,
        mag: float,
        template: str,
        magtype: str = "Vega",
        band: str = "Cousins_I.dat",
        cwave: str = "7000",
        slitwidth: str = "0.75",
        binning: str = "1x1",
        airmass: float = 1.1,
        seeing: float = 0.75,
        redshift: float = 0,
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
        if grating not in wmko_options["grating (DEIMOS)"]:
            raise KeyError(f"{grating} not one of {wmko_options['grating (DEIMOS)']}")
        if binning not in wmko_options["binning (DEIMOS)"]:
            raise KeyError(f"{binning} not one of {wmko_options['binning (DEIMOS)']}")
        if slitwidth not in wmko_options["slitwidth (DEIMOS)"]:
            raise KeyError(
                f"{slitwidth} not one of {wmko_options['slitwidth (DEIMOS)']}"
            )
        if cwave not in wmko_options["central wavelength (DEIMOS)"]:
            raise KeyError(
                f"{cwave} not one of {wmko_options['central wavelength (DEIMOS)']}"
            )
        self.grating = grating
        self.binning = binning
        self.slitwidth = slitwidth
        self.cwave = cwave

    def query_s2n(self):
        """
        Query the DEIMOS ETC (http://etc.ucolick.org/web_s2n/deimos)

        :return:
        """
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
        return snr


class Sig2NoiseLRIS(Sig2NoiseWMKO):
    """
    Keck/LRIS S/N Query (http://etc.ucolick.org/web_s2n/lris)

    :param str grating: LRIS red arm grating.
        Must be one of "600/7500", "600/10000", "1200/9000", "400/8500", or "831/8200".
    :param str grism: LRIS blue arm grism. Must be one of "B300" or "B600".
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.wmko_options['template'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str band: Magnitude band. For valid options see s2n.wmko_options['filter'].
    :param str dichroic: LRIS dichroic separating the red and blue arms. "D560" is the only option currently.
    :param str slitwidth: Width of slit in arcseconds. Must be one of "0.7", "1.0", or "1.5"
    :param str binning: spatial x spectral binning. Must be one of "1x1", "2x1", "2x2", or "3x1"
    :param float airmass: Airmass of observation
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float redshift: Redshift of the target
    """

    def __init__(
        self,
        grating: str,
        grism: str,
        exptime: float,
        mag: float,
        template: str,
        magtype: str = "Vega",
        band: str = "Cousins_I.dat",
        dichroic: str = "D560",
        slitwidth: str = "0.7",
        binning: str = "1x1",
        airmass: float = 1.1,
        seeing: float = 0.75,
        redshift: float = 0,
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
        if grating not in wmko_options["grating (LRIS)"]:
            raise KeyError(f"{grating} not one of {wmko_options['grating (LRIS)']}")
        if grism not in wmko_options["grism (LRIS)"]:
            raise KeyError(f"{grism} not one of {wmko_options['grism (LRIS)']}")
        if binning not in wmko_options["binning (LRIS)"]:
            raise KeyError(f"{binning} not one of {wmko_options['binning (LRIS)']}")
        if slitwidth not in wmko_options["slitwidth (LRIS)"]:
            raise KeyError(f"{slitwidth} not one of {wmko_options['slitwidth (LRIS)']}")
        if dichroic not in wmko_options["dichroic (LRIS)"]:
            raise KeyError(f"{dichroic} not one of {wmko_options['dichroic (LRIS)']}")
        self.grating = grating
        self.grism = grism
        self.binning = binning
        self.slitwidth = slitwidth
        self.dichroic = dichroic

    def query_s2n(self):
        """
        Query the LRIS ETC (http://etc.ucolick.org/web_s2n/lris)

        :return:
        """
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
        return snr


class Sig2NoiseESI(Sig2NoiseWMKO):
    """
    Keck/ESI S/N Query (http://etc.ucolick.org/web_s2n/esi)

    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.wmko_options['template'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str band: Magnitude band. For valid options see s2n.wmko_options['filter'].
    :param str slitwidth: Width of slit in arcseconds. Must be one of "0.75", "0.3", "0.5", or "1.0"
    :param str binning: spatial x spectral binning. Must be one of "1x1", "2x1", "2x2", or "3x1"
    :param float airmass: Airmass of observation
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float redshift: Redshift of the target
    """

    def __init__(
        self,
        exptime: float,
        mag: float,
        template: str,
        magtype: str = "Vega",
        band: str = "Cousins_I.dat",
        slitwidth: str = "0.75",
        binning: str = "1x1",
        airmass: float = 1.1,
        seeing: float = 0.75,
        redshift: float = 0,
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
        if binning not in wmko_options["binning (ESI)"]:
            raise KeyError(f"{binning} not one of {wmko_options['binning (ESI)']}")
        if slitwidth not in wmko_options["slitwidth (ESI)"]:
            raise KeyError(f"{slitwidth} not one of {wmko_options['slitwidth (ESI)']}")
        self.binning = binning
        self.slitwidth = slitwidth

    def query_s2n(self):
        """
        Query the ESI ETC (http://etc.ucolick.org/web_s2n/esi)

        :return:
        """
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
        return snr


class Sig2NoiseHIRES(Sig2NoiseWMKO):
    """
    Keck/HIRES S/N Query (http://etc.ucolick.org/web_s2n/hires)

    :param str slitwidth: HIRES Decker. Must be "C5" (1.15"), "E4" (0.40"), "B2" (0.57"),
                                                "B5" (0.86"), "E5" (0.80"), or "D3" (1.72")
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.wmko_options['template'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str band: Magnitude band. For valid options see s2n.wmko_options['filter'].
    :param str binning: spatial x spectral binning. Must be one of "1x1", "2x1", "2x2", or "3x1".
    :param float airmass: Airmass of observation
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float redshift: Redshift of the target
    """

    def __init__(
        self,
        slitwidth: str,
        exptime: float,
        mag: float,
        template: str,
        magtype: str = "Vega",
        band: str = "Cousins_I.dat",
        binning: str = "1x1",
        airmass: float = 1.1,
        seeing: float = 0.75,
        redshift: float = 0,
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
        if binning not in wmko_options["binning (HIRES)"]:
            raise KeyError(f"{binning} not one of {wmko_options['binning (HIRES)']}")
        if slitwidth not in wmko_options["slitwidth (HIRES)"]:
            raise KeyError(
                f"{slitwidth} not one of {wmko_options['slitwidth (HIRES)']}"
            )
        self.binning = binning
        self.slitwidth = slitwidth

    def query_s2n(self):
        """
        Query the HIRES ETC (http://etc.ucolick.org/web_s2n/hires)

        :return:
        """
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
        return snr


class Sig2NoiseHectoBinoSpec(Sig2NoiseQuery):
    """
    MMT/Hectospec and MMT/Binospec S/N Query (http://hopper.si.edu/etc-cgi/TEST/sao-etc)

    :param str inst_mode: Instrument and mode.
                          One of: "BINOSPEC_1000", "BINOSPEC_270", "BINOSPEC_600", "HECTOSPEC_270", or "HECTOSPEC_600"
    :param float exptime: Exposure time in seconds
    :param float mag: AB Magnitude of source
    :param str band: Magnitude band. One of "r_filt", "g_filt", or "i_filt"
    :param str template: Spectral template. For valid options see s2n.mmt_options['template'].
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float airmass: Airmass of observation
    :param float moonage: Moon Phase (days since new moon)
    :param str aptype: Aperture shape. Must be one of "Round", "Square", or "Rectangular".
    :param float apwidth: Width of aperture in arcseconds
    """

    def __init__(
        self,
        inst_mode: str,
        exptime: float,
        mag: float,
        band: str = "g_filt",
        template: str = "K0V",
        seeing: float = 0.75,
        airmass: float = 1.1,
        moonage: float = 0.0,
        aptype: str = "Round",
        apwidth: float = 1.0,
    ):
        Sig2NoiseQuery.__init__(self)
        if inst_mode not in mmt_options["inst_mode"]:
            raise KeyError(f"{inst_mode} not one of {mmt_options['inst_mode']}")
        self.inst_mode = inst_mode
        self.exptime = exptime
        self.mag = mag
        if band not in mmt_options["filter"]:
            raise KeyError(f"{band} not one of {mmt_options['filter']}")
        self.band = band
        if template not in mmt_options["template"]:
            raise KeyError(f"{template} not one of {mmt_options['template']}")
        self.template = template
        self.seeing = seeing
        self.airmass = airmass
        self.moonage = moonage
        if aptype not in mmt_options["aptype"]:
            raise KeyError(f"{aptype} not one of {mmt_options['aptype']}")
        self.aptype = aptype
        self.apwidth = apwidth

    def query_s2n(self):
        """
        Query the Hectospec/Binospec ETC (http://hopper.si.edu/etc-cgi/TEST/sao-etc)

        :return:
        """
        url = "http://hopper.si.edu/etc-cgi/TEST/sao-etc"
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="instmode", value="")
        form.new_control(type="select", name="objspec_", value="")
        form.new_control(type="select", name="objspec__", value="")
        form["instmode"] = self.inst_mode
        form["exptime"] = self.exptime
        form["ABmag"] = self.mag
        form["bandfilter"] = self.band
        form["objspec_"] = "Stars"
        form["objspec__"] = self.template
        form["objspec"] = f"Stars/{self.template}.tab"
        form["srcext"] = 0.0
        form["seeing"] = self.seeing
        form["airmass"] = self.airmass
        form["moonage"] = self.moonage
        form["aptype"] = self.aptype
        form["apwidth"] = self.apwidth
        data = browser.submit_selected()
        snr_text = data.text.split("---")[-1]
        snr = pd.DataFrame([row.split("\t") for row in snr_text.split("\n")[1:-1]])
        snr.index = snr.pop(0)
        snr.drop([1, 2, 3, 4], axis=1, inplace=True)
        snr = np.vstack([snr.index.values, snr[5].values]).astype(float)
        snr[0] *= 1e4
        return snr


class Sig2NoiseVLT(Sig2NoiseQuery):
    """
    Superclass for VLT ETC Queries

    :param str instrument: VLT instrument. Must be "UVES", "FLAMES-UVES", "FLAMES-GIRAFFE", "X-SHOOTER", or "MUSE"
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (<instrument>)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    # TODO: Implement MARCS stellar template selection
    def __init__(
        self,
        instrument: str,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        **kwargs,
    ):
        Sig2NoiseQuery.__init__(self)
        if instrument not in vlt_options["instruments"]:
            raise KeyError(f"{instrument} not one of {vlt_options['instruments']}")
        if not exptime > 0:
            raise ValueError("Exposure Time must be positive")
        if magtype not in vlt_options["src_target_mag_system"]:
            raise KeyError(
                f"{magtype} not one of {vlt_options['src_target_mag_system']}"
            )
        if template_type not in vlt_options["src_target_type"]:
            raise KeyError(
                f"{template_type} not one of {vlt_options['src_target_type']}"
            )
        if template not in vlt_options["src_target_spec_type"]:
            raise KeyError(
                f"{template} not one of {vlt_options['src_target_spec_type']}"
            )
        if not redshift >= 0:
            raise ValueError("Redshift must be positive")
        if not airmass >= 1.0:
            raise ValueError("Airmass must be > 1.0")
        if moon_phase < 0.0 or moon_phase > 1.0:
            raise ValueError("moon_phase must be between 0.0 (new) and 1.0 (full)")
        if seeing not in vlt_options["sky_seeing"]:
            raise KeyError(f"{seeing} not one of {vlt_options['sky_seeing']}")
        self.instrument = instrument
        self.exptime = exptime
        self.mag = mag
        self.band = band
        self.magtype = magtype
        self.template_type = template_type
        self.template = template
        self.redshift = redshift
        self.airmass = airmass
        self.moon_phase = moon_phase
        self.seeing = seeing
        self.kwargs = kwargs

    def query_s2n(self) -> None:
        """
        No generic S/N query, see specific instrument subclasses

        :return:
        """
        raise NotImplementedError(
            "No generic S/N query, see specific instrument children classes"
        )


class Sig2NoiseUVES(Sig2NoiseVLT):
    """
    VLT/UVES S/N Query (http://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES++INS.MODE=spectro)

    :param str detector: UVES detector setup. For valid options see s2n.vlt_options['uves_det_cd_name'].
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (UVES)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param str slitwidth: Width of slit in arcseconds. For valid options see s2n.vlt_options['uves_slit_width'].
    :param str binning: spatial x spectral binning. For valid options see s2n.vlt_options['uves_ccd_binning'].
    :param bool mid_order_only: If True, returns only peak S/N in each order.
        Otherwise the S/N at both ends of each order are also included.
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    def __init__(
        self,
        detector: str,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        slitwidth: str = "1.0",
        binning: str = "1x1",
        mid_order_only: bool = False,
        **kwargs,
    ):
        Sig2NoiseVLT.__init__(
            self,
            "UVES",
            exptime,
            mag,
            band,
            magtype,
            template_type,
            template,
            redshift,
            airmass,
            moon_phase,
            seeing,
            **kwargs,
        )
        self.url = "http://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES++INS.MODE=spectro"
        if self.band not in vlt_options["src_target_mag_band (UVES)"]:
            raise KeyError(
                f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band (UVES)']}"
            )
        if detector not in vlt_options["uves_det_cd_name"]:
            raise KeyError(f"{detector} not one of {vlt_options['uves_det_cd_name']}")
        if slitwidth not in vlt_options["uves_slit_width"]:
            raise KeyError(f"{slitwidth} not one of {vlt_options['uves_slit_width']}")
        if binning not in vlt_options["uves_ccd_binning"]:
            raise KeyError(f"{binning} not one of {vlt_options['uves_ccd_binning']}")
        self.detector = detector
        self.slitwidth = slitwidth
        self.binning = binning
        self.mid_order_only = mid_order_only
        self.data = None

    def query_s2n(self):
        """
        Query the UVES ETC (http://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES++INS.MODE=spectro)

        :return:
        """
        url = self.url
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="SRC.TARGET.MAG.BAND", value="")
        form.new_control(type="select", name="SKY.SEEING.ZENITH.V", value="")
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.mag
        form["SRC.TARGET.MAG.BAND"] = self.band
        form["SRC.TARGET.MAG.SYSTEM"] = self.magtype
        form["SRC.TARGET.TYPE"] = self.template_type
        form["SRC.TARGET.SPEC.TYPE"] = self.template
        form["SRC.TARGET.REDSHIFT"] = self.redshift
        form["SRC.TARGET.GEOM"] = "seeing_ltd"
        # Sky Parameters
        form["SKY.AIRMASS"] = self.airmass
        form["SKY.MOON.FLI"] = self.moon_phase
        form["USR.SEEING.OR.IQ"] = "seeing_given"
        form["SKY.SEEING.ZENITH.V"] = self.seeing
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        # Instrument Specifics
        form["INS.NAME"] = "UVES"
        form["INS.MODE"] = "spectro"
        form["INS.PRE_SLIT.FILTER.NAME"] = "ADC"
        form["INS.IMAGE_SLICERS.NAME"] = "None"
        form["INS.BELOW_SLIT.FILTER.NAME"] = "NONE"
        form["INS.DET.SPECTRAL_FORMAT.NAME"] = "STANDARD"
        form["INS.DET.CD.NAME"] = self.detector
        form["INS.SLIT.FROM_USER.WIDTH.VAL"] = self.slitwidth
        form["INS.DET.CCD.BINNING.VAL"] = self.binning
        form["INS.DET.EXP.TIME.VAL"] = self.exptime
        form["INS.GEN.TABLE.SF.SWITCH.VAL"] = "yes"
        form["INS.GEN.TABLE.RES.SWITCH.VAL"] = "yes"
        form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        for key in self.kwargs:
            form[key] = self.kwargs[key]
        self.data = browser.submit_selected()
        if self.mid_order_only:
            snr = self.parse_etc_mid()
        else:
            snr = self.parse_etc()
        return snr

    def parse_etc(self):
        mit_tab1 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[1].split("</table>")[0]
        )[0]
        mit_tab1.columns = mit_tab1.loc[0]
        mit_tab1.drop(0, axis=0, inplace=True)
        mit_tab2 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[2].split("</table>")[0]
        )[0]
        mit_tab2.columns = mit_tab2.loc[1]
        mit_tab2.drop([0, 1], axis=0, inplace=True)
        eev_tab1 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[3].split("</table>")[0]
        )[0]
        eev_tab1.columns = eev_tab1.loc[0]
        eev_tab1.drop(0, axis=0, inplace=True)
        eev_tab2 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[4].split("</table>")[0]
        )[0]
        eev_tab2.columns = eev_tab2.loc[1]
        eev_tab2.drop([0, 1], axis=0, inplace=True)
        mit_wave_mid = mit_tab1["wav of central column (nm)"]
        mit_wave_min = mit_tab1["FSR l Min (nm)"]
        mit_wave_max = mit_tab1["FSR l Max (nm)"]
        mit_snr_min = mit_tab2["S/N*"].iloc[:, 0]
        mit_snr_mid = mit_tab2["S/N*"].iloc[:, 1]
        mit_snr_max = mit_tab2["S/N*"].iloc[:, 2]
        eev_wave_mid = eev_tab1["wav of central column (nm)"]
        eev_wave_min = eev_tab1["FSR l Min (nm)"]
        eev_wave_max = eev_tab1["FSR l Max (nm)"]
        eev_snr_min = eev_tab2["S/N*"].iloc[:, 0]
        eev_snr_mid = eev_tab2["S/N*"].iloc[:, 1]
        eev_snr_max = eev_tab2["S/N*"].iloc[:, 2]
        mit_wave = pd.concat([mit_wave_min, mit_wave_mid, mit_wave_max])
        mit_snr = pd.concat([mit_snr_min, mit_snr_mid, mit_snr_max])
        mit_snr.index = mit_wave
        mit_snr.sort_index(inplace=True)
        mit_snr = mit_snr.groupby(mit_snr.index).max()
        eev_wave = pd.concat([eev_wave_min, eev_wave_mid, eev_wave_max])
        eev_snr = pd.concat([eev_snr_min, eev_snr_mid, eev_snr_max])
        eev_snr.index = eev_wave
        eev_snr.sort_index(inplace=True)
        eev_snr = eev_snr.groupby(eev_snr.index).max()
        uves_snr = pd.concat([eev_snr, mit_snr])
        uves_snr = np.vstack([uves_snr.index.values, uves_snr.iloc[:].values]).astype(
            float
        )
        uves_snr[0] *= 10
        return uves_snr

    def parse_etc_mid(self):
        snr_url1 = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        )
        snr_url2 = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[2].split('" TITLE')[0]
        )
        snr_txt1 = requests.post(snr_url1).text
        snr_txt2 = requests.post(snr_url2).text
        snr1 = pd.DataFrame([row.split("\t") for row in snr_txt1.split("\n")[:-1]])
        snr2 = pd.DataFrame([row.split("\t") for row in snr_txt2.split("\n")[:-1]])
        uves_snr = pd.concat([snr1, snr2])
        uves_snr.index = uves_snr.pop(0)
        uves_snr.sort_index(inplace=True)
        uves_snr = np.vstack([uves_snr.index.values, uves_snr[1].values]).astype(float)
        uves_snr[0] *= 10
        return uves_snr


class Sig2NoiseFLAMESUVES(Sig2NoiseVLT):
    """
    VLT/FLAMES-UVES S/N Query (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES+INS.MODE=FLAMES)

    :param str detector: UVES detector setup. For valid options see s2n.vlt_options['uves_det_cd_name'].
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (UVES)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param bool mid_order_only: If True, returns only peak S/N in each order.
        Otherwise the S/N at both ends of each order are also included.
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    def __init__(
        self,
        detector: str,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        mid_order_only: bool = False,
        **kwargs,
    ):
        Sig2NoiseVLT.__init__(
            self,
            "FLAMES-UVES",
            exptime,
            mag,
            band,
            magtype,
            template_type,
            template,
            redshift,
            airmass,
            moon_phase,
            seeing,
            **kwargs,
        )
        self.url = "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES+INS.MODE=FLAMES"
        if self.band not in vlt_options["src_target_mag_band (UVES)"]:
            raise KeyError(
                f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band (UVES)']}"
            )
        if detector not in vlt_options["uves_det_cd_name"]:
            raise KeyError(f"{detector} not one of {vlt_options['uves_det_cd_name']}")
        self.detector = detector
        self.mid_order_only = mid_order_only
        self.data = None

    def query_s2n(self):
        """
        Query the FLAMES-UVES ETC (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES+INS.MODE=FLAMES)

        :return:
        """
        url = self.url
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="SRC.TARGET.MAG.BAND", value="")
        form.new_control(type="select", name="SKY.SEEING.ZENITH.V", value="")
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.mag
        form["SRC.TARGET.MAG.BAND"] = self.band
        form["SRC.TARGET.MAG.SYSTEM"] = self.magtype
        form["SRC.TARGET.TYPE"] = self.template_type
        form["SRC.TARGET.SPEC.TYPE"] = self.template
        form["SRC.TARGET.REDSHIFT"] = self.redshift
        form["SRC.TARGET.GEOM"] = "seeing_ltd"
        # Sky Parameters
        form["SKY.AIRMASS"] = self.airmass
        form["SKY.MOON.FLI"] = self.moon_phase
        form["USR.SEEING.OR.IQ"] = "seeing_given"
        form["SKY.SEEING.ZENITH.V"] = self.seeing
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        # Instrument Specifics
        form["INS.NAME"] = "UVES"
        form["INS.MODE"] = "FLAMES"
        form["INS.DET.CD.NAME"] = self.detector
        form["INS.DET.EXP.TIME.VAL"] = self.exptime
        form["INS.GEN.TABLE.SF.SWITCH.VAL"] = "yes"
        form["INS.GEN.TABLE.RES.SWITCH.VAL"] = "yes"
        form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        for key in self.kwargs:
            form[key] = self.kwargs[key]
        self.data = browser.submit_selected()
        if self.mid_order_only:
            snr = self.parse_etc_mid()
        else:
            snr = self.parse_etc()
        return snr

    def parse_etc(self):
        mit_tab1 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[1].split("</table>")[0]
        )[0]
        mit_tab1.columns = mit_tab1.loc[0]
        mit_tab1.drop(0, axis=0, inplace=True)
        mit_tab2 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[2].split("</table>")[0]
        )[0]
        mit_tab2.columns = mit_tab2.loc[1]
        mit_tab2.drop([0, 1], axis=0, inplace=True)
        eev_tab1 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[3].split("</table>")[0]
        )[0]
        eev_tab1.columns = eev_tab1.loc[0]
        eev_tab1.drop(0, axis=0, inplace=True)
        eev_tab2 = pd.read_html(
            '<table class="echelleTable'
            + self.data.text.split('<table class="echelleTable')[4].split("</table>")[0]
        )[0]
        eev_tab2.columns = eev_tab2.loc[1]
        eev_tab2.drop([0, 1], axis=0, inplace=True)
        mit_wave_mid = mit_tab1["wav of central column (nm)"]
        mit_wave_min = mit_tab1["FSR l Min (nm)"]
        mit_wave_max = mit_tab1["FSR l Max (nm)"]
        mit_snr_min = mit_tab2["S/N*"].iloc[:, 0]
        mit_snr_mid = mit_tab2["S/N*"].iloc[:, 1]
        mit_snr_max = mit_tab2["S/N*"].iloc[:, 2]
        eev_wave_mid = eev_tab1["wav of central column (nm)"]
        eev_wave_min = eev_tab1["FSR l Min (nm)"]
        eev_wave_max = eev_tab1["FSR l Max (nm)"]
        eev_snr_min = eev_tab2["S/N*"].iloc[:, 0]
        eev_snr_mid = eev_tab2["S/N*"].iloc[:, 1]
        eev_snr_max = eev_tab2["S/N*"].iloc[:, 2]
        mit_wave = pd.concat([mit_wave_min, mit_wave_mid, mit_wave_max])
        mit_snr = pd.concat([mit_snr_min, mit_snr_mid, mit_snr_max])
        mit_snr.index = mit_wave
        mit_snr.sort_index(inplace=True)
        mit_snr = mit_snr.groupby(mit_snr.index).max()
        eev_wave = pd.concat([eev_wave_min, eev_wave_mid, eev_wave_max])
        eev_snr = pd.concat([eev_snr_min, eev_snr_mid, eev_snr_max])
        eev_snr.index = eev_wave
        eev_snr.sort_index(inplace=True)
        eev_snr = eev_snr.groupby(eev_snr.index).max()
        uves_snr = pd.concat([eev_snr, mit_snr])
        uves_snr = np.vstack([uves_snr.index.values, uves_snr.iloc[:].values]).astype(
            float
        )
        uves_snr[0] *= 10
        return uves_snr

    def parse_etc_mid(self):
        snr_url1 = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        )
        snr_url2 = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[2].split('" TITLE')[0]
        )
        snr_txt1 = requests.post(snr_url1).text
        snr_txt2 = requests.post(snr_url2).text
        snr1 = pd.DataFrame([row.split("\t") for row in snr_txt1.split("\n")[:-1]])
        snr2 = pd.DataFrame([row.split("\t") for row in snr_txt2.split("\n")[:-1]])
        uves_snr = pd.concat([snr1, snr2])
        uves_snr.index = uves_snr.pop(0)
        uves_snr.sort_index(inplace=True)
        uves_snr = np.vstack([uves_snr.index.values, uves_snr[1].values]).astype(float)
        uves_snr[0] *= 10
        return uves_snr


class Sig2NoiseFLAMESGIRAFFE(Sig2NoiseVLT):
    """
    VLT/FLAMES-GIRAFFE S/N Query (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=GIRAFFE+INS.MODE=spectro)

    :param str slicer: GIRAFFE slicer. For valid options see s2n.vlt_options['giraffe_slicer'].
    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (UVES)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param str sky_sampling_mode: Fiber Mode. Must be one of "MEDUSA", "IFU052", "ARGUS052", or "ARGUS030".
    :param str ccd_mode: CCD readout mode. Must be one of "standard", "fast", or "slow"
    :param float fiber_obj_decenter: Displacement of source from fiber center (from 0.0 to 0.6).
        Only applicable if sky_sampling_mode="MEDUSA".
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    def __init__(
        self,
        slicer: str,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        sky_sampling_mode="MEDUSA",
        ccd_mode="standard",
        fiber_obj_decenter=0.0,
        **kwargs,
    ):
        Sig2NoiseVLT.__init__(
            self,
            "FLAMES-GIRAFFE",
            exptime,
            mag,
            band,
            magtype,
            template_type,
            template,
            redshift,
            airmass,
            moon_phase,
            seeing,
            **kwargs,
        )
        self.url = "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=GIRAFFE+INS.MODE=spectro"
        if self.band not in vlt_options["src_target_mag_band (GIRAFFE)"]:
            raise KeyError(
                f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band (GIRAFFE)']}"
            )
        if slicer not in vlt_options["giraffe_slicer"]:
            raise KeyError(f"{slicer} not one of {vlt_options['giraffe_slicer']}")
        if sky_sampling_mode not in vlt_options["giraffe_sky_sampling_mode"]:
            raise KeyError(
                f"{sky_sampling_mode} not one of {vlt_options['giraffe_sky_sampling_mode']}"
            )
        if ccd_mode not in vlt_options["giraffe_ccd_mode"]:
            raise KeyError(f"{ccd_mode} not one of {vlt_options['giraffe_ccd_mode']}")
        if not fiber_obj_decenter >= 0:
            raise ValueError("giraffe_fiber_obj_decenter must be positive")
        self.slicer = slicer
        self.sky_sampling_mode = sky_sampling_mode
        self.ccd_mode = ccd_mode
        self.fiber_obj_decenter = fiber_obj_decenter
        self.data = None

    def query_s2n(self):
        """
        Query the FLAMES-GIRAFFE ETC (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=GIRAFFE+INS.MODE=spectro)

        :return:
        """
        url = self.url
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="SRC.TARGET.MAG.BAND", value="")
        form.new_control(type="select", name="SKY.SEEING.ZENITH.V", value="")
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.mag
        form["SRC.TARGET.MAG.BAND"] = self.band
        form["SRC.TARGET.MAG.SYSTEM"] = self.magtype
        form["SRC.TARGET.TYPE"] = self.template_type
        form["SRC.TARGET.SPEC.TYPE"] = self.template
        form["SRC.TARGET.REDSHIFT"] = self.redshift
        form["SRC.TARGET.GEOM"] = "seeing_ltd"
        # Sky Parameters
        form["SKY.AIRMASS"] = self.airmass
        form["SKY.MOON.FLI"] = self.moon_phase
        form["USR.SEEING.OR.IQ"] = "seeing_given"
        form["SKY.SEEING.ZENITH.V"] = self.seeing
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        # Instrument Specifics
        form["INS.NAME"] = "GIRAFFE"
        form["INS.MODE"] = "spectro"
        form["INS.SKY.SAMPLING.MODE"] = self.sky_sampling_mode
        form["INS.GIRAFFE.FIBER.OBJ.DECENTER"] = self.fiber_obj_decenter
        if self.slicer[:2] == "LR":
            form["INS.GIRAFFE.RESOLUTION"] = "LR"
            form["INS.IMAGE.SLICERS.NAME.LR"] = self.slicer
        elif self.slicer[:2] == "HR":
            form["INS.GIRAFFE.RESOLUTION"] = "HR"
            form["INS.IMAGE.SLICERS.NAME.HR"] = self.slicer
        else:
            raise RuntimeError(f"{self.slicer} should start with either 'LR' or 'HR'")
        form["DET.CCD.MODE"] = self.ccd_mode
        form["USR.OUT.MODE"] = "USR.OUT.MODE.EXPOSURE.TIME"
        form["USR.OUT.MODE.EXPOSURE.TIME"] = self.exptime
        form["USR.OUT.DISPLAY.SN.V.WAVELENGTH"] = "1"
        for key in self.kwargs:
            form[key] = self.kwargs[key]
        self.data = browser.submit_selected()
        snr = self.parse_etc()
        return snr

    def parse_etc(self):
        snr_url = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        )
        snr_txt = requests.post(snr_url).text
        snr = pd.DataFrame([row.split(" ") for row in snr_txt.split("\n")[:-1]])
        snr.index = snr.pop(0)
        snr.sort_index(inplace=True)
        snr = np.vstack([snr.index.values, snr[1].values]).astype(float)
        snr[0] *= 10
        return snr


class Sig2NoiseXSHOOTER(Sig2NoiseVLT):
    """
    VLT/X-SHOOTER S/N Query (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=X-SHOOTER+INS.MODE=spectro)

    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (UVES)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param str uvb_slitwidth: Width of UVB spectrograph slit in arcseconds
    :param str vis_slitwidth: Width of VIS spectrograph slit in arcseconds
    :param str nir_slitwidth: Width of NIR spectrograph slit in arcseconds
    :param str uvb_ccd_binning: UVB CCD gain/binning/readout mode.
        For valid options see s2n.vlt_options['xshooter_uvb_ccd_binning'].
    :param str vis_ccd_binning: VIS CCD gain/binning/readout mode.
        For valid options see s2n.vlt_options['xshooter_vis_ccd_binning'].
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    def __init__(
        self,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        uvb_slitwidth: str = "0.8",
        vis_slitwidth: str = "0.7",
        nir_slitwidth: str = "0.9",
        uvb_ccd_binning: str = "high1x1slow",
        vis_ccd_binning: str = "high1x1slow",
        **kwargs,
    ):
        Sig2NoiseVLT.__init__(
            self,
            "X-SHOOTER",
            exptime,
            mag,
            band,
            magtype,
            template_type,
            template,
            redshift,
            airmass,
            moon_phase,
            seeing,
            **kwargs,
        )
        self.url = "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=X-SHOOTER+INS.MODE=spectro"
        if self.band not in vlt_options["src_target_mag_band (X-SHOOTER)"]:
            raise KeyError(
                f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band (X-SHOOTER)']}"
            )
        if uvb_slitwidth not in vlt_options["xshooter_uvb_slitwidth"]:
            raise KeyError(
                f"{uvb_slitwidth} not one of {vlt_options['xshooter_uvb_slitwidth']}"
            )
        if vis_slitwidth not in vlt_options["xshooter_vis_slitwidth"]:
            raise KeyError(
                f"{vis_slitwidth} not one of {vlt_options['xshooter_vis_slitwidth']}"
            )
        if nir_slitwidth not in vlt_options["xshooter_nir_slitwidth"]:
            raise KeyError(
                f"{nir_slitwidth} not one of {vlt_options['xshooter_nir_slitwidth']}"
            )
        if uvb_ccd_binning not in vlt_options["xshooter_uvb_ccd_binning"]:
            raise KeyError(
                f"{uvb_ccd_binning} not one of {vlt_options['xshooter_uvb_ccd_binning']}"
            )
        if vis_ccd_binning not in vlt_options["xshooter_vis_ccd_binning"]:
            raise KeyError(
                f"{vis_ccd_binning} not one of {vlt_options['xshooter_vis_ccd_binning']}"
            )
        self.uvb_slitwidth = uvb_slitwidth
        self.vis_slitwidth = vis_slitwidth
        self.nir_slitwidth = nir_slitwidth
        self.uvb_ccd_binning = uvb_ccd_binning
        self.vis_ccd_binning = vis_ccd_binning
        self.data = None

    def query_s2n(self):
        """
        Query the X-SHOOTER ETC (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=X-SHOOTER+INS.MODE=spectro)

        :return:
        """
        url = self.url
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="SRC.TARGET.MAG.BAND", value="")
        form.new_control(type="select", name="SKY.SEEING.ZENITH.V", value="")
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.mag
        form["SRC.TARGET.MAG.BAND"] = self.band
        form["SRC.TARGET.MAG.SYSTEM"] = self.magtype
        form["SRC.TARGET.TYPE"] = self.template_type
        form["SRC.TARGET.SPEC.TYPE"] = self.template
        form["SRC.TARGET.REDSHIFT"] = self.redshift
        form["SRC.TARGET.GEOM"] = "seeing_ltd"
        # Sky Parameters
        form["SKY.AIRMASS"] = self.airmass
        form["SKY.MOON.FLI"] = self.moon_phase
        form["USR.SEEING.OR.IQ"] = "seeing_given"
        form["SKY.SEEING.ZENITH.V"] = self.seeing
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        # Instrument Specifics
        form["INS.NAME"] = "X-SHOOTER"
        form["INS.MODE"] = "spectro"
        form["INS.ARM.UVB.FLAG"] = "1"
        form["INS.ARM.VIS.FLAG"] = "1"
        form["INS.ARM.NIR.FLAG"] = "1"
        form["INS.SLIT.FROM_USER.WIDTH.VAL.UVB"] = self.uvb_slitwidth
        form["INS.SLIT.FROM_USER.WIDTH.VAL.VIS"] = self.vis_slitwidth
        form["INS.SLIT.FROM_USER.WIDTH.VAL.NIR"] = self.nir_slitwidth
        form["INS.DET.DIT.UVB"] = self.exptime
        form["INS.DET.DIT.VIS"] = self.exptime
        form["INS.DET.DIT.NIR"] = self.exptime
        form["INS.DET.CCD.BINNING.VAL.UVB"] = self.uvb_ccd_binning
        form["INS.DET.CCD.BINNING.VAL.VIS"] = self.vis_ccd_binning
        form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        for key in self.kwargs:
            form[key] = self.kwargs[key]
        self.data = browser.submit_selected()
        snr = self.parse_etc()
        return snr

    def parse_etc(self):
        def combine_xshooter_snr(snr_min_df, snr_mid_df, snr_max_df, offset):
            snr_mid_df.index = snr_mid_df.pop(0)
            snr_max_df.index = snr_max_df.pop(0)
            snr_min_df.index = snr_min_df.pop(0)
            snr_mid_df.sort_index(inplace=True)
            snr_max_df.sort_index(inplace=True)
            snr_min_df.sort_index(inplace=True)
            snr = pd.concat([snr_min_df, snr_mid_df, snr_max_df])
            snr.sort_index(inplace=True)
            for i, idx_min in enumerate(snr_min_df.index[offset:]):
                idx_max = snr_max_df.index[i]
                idx_mid_before = snr_mid_df.index[i]
                idx_mid_after = snr_mid_df.index[i + 1]
                snr_min = snr_min_df.loc[idx_min, 1]
                if not isinstance(snr_min, float):
                    snr_min = snr_min.iloc[0]
                snr_mid_before = snr_mid_df.loc[idx_mid_before, 1]
                snr_mid_after = snr_mid_df.loc[idx_mid_after, 1]
                snr_max = snr_max_df.loc[idx_max, 1]
                if not isinstance(snr_max, float):
                    snr_max = snr_max.iloc[0]
                if idx_min < idx_max:
                    if snr_min > snr_max:
                        dy = snr_max - snr_mid_before
                        dx = idx_max - idx_mid_before
                        new_wave = idx_min - 0.1
                        new_snr = snr_mid_before + dy / dx * (new_wave - idx_mid_before)
                        snr.drop(idx_max, inplace=True)
                        snr.loc[new_wave] = new_snr
                    else:
                        dy = snr_mid_after - snr_min
                        dx = idx_mid_after - idx_min
                        new_wave = idx_max + 0.1
                        new_snr = snr_min + dy / dx * (new_wave - idx_min)
                        snr.drop(idx_min, inplace=True)
                        snr.loc[new_wave] = new_snr
                elif idx_min == idx_max:
                    snr.drop(idx_min, inplace=True)
                    snr.loc[idx_min] = np.max([snr_min, snr_max])
                snr.sort_index(inplace=True)
            return snr

        snr_url1_mid = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        )
        snr_url2_mid = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[2].split('" TITLE')[0]
        )
        snr_url3_mid = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[3].split('" TITLE')[0]
        )
        snr_url1_max = snr_url1_mid[:-4] + "_FSRmax.dat"
        snr_url2_max = snr_url2_mid[:-4] + "_FSRmax.dat"
        snr_url3_max = snr_url3_mid[:-4] + "_FSRmax.dat"
        snr_url1_min = snr_url1_mid[:-4] + "_FSRmin.dat"
        snr_url2_min = snr_url2_mid[:-4] + "_FSRmin.dat"
        snr_url3_min = snr_url3_mid[:-4] + "_FSRmin.dat"
        snr_txt1_mid = requests.post(snr_url1_mid).text
        snr_txt2_mid = requests.post(snr_url2_mid).text
        snr_txt3_mid = requests.post(snr_url3_mid).text
        snr_txt1_max = requests.post(snr_url1_max).text
        snr_txt2_max = requests.post(snr_url2_max).text
        snr_txt3_max = requests.post(snr_url3_max).text
        snr_txt1_min = requests.post(snr_url1_min).text
        snr_txt2_min = requests.post(snr_url2_min).text
        snr_txt3_min = requests.post(snr_url3_min).text
        snr1_mid_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt1_mid.split("\n")[:-1]], dtype="float64"
        )
        snr2_mid_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt2_mid.split("\n")[:-1]], dtype="float64"
        )
        snr3_mid_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt3_mid.split("\n")[:-1]], dtype="float64"
        )
        snr1_max_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt1_max.split("\n")[:-1]], dtype="float64"
        )
        snr2_max_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt2_max.split("\n")[:-1]], dtype="float64"
        )
        snr3_max_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt3_max.split("\n")[:-1]], dtype="float64"
        )
        snr1_min_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt1_min.split("\n")[:-1]], dtype="float64"
        )
        snr2_min_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt2_min.split("\n")[:-1]], dtype="float64"
        )
        snr3_min_df = pd.DataFrame(
            [row.split("\t") for row in snr_txt3_min.split("\n")[:-1]], dtype="float64"
        )
        snr1 = combine_xshooter_snr(snr1_min_df, snr1_mid_df, snr1_max_df, offset=1)
        snr2 = combine_xshooter_snr(snr2_min_df, snr2_mid_df, snr2_max_df, offset=0)
        snr3 = combine_xshooter_snr(snr3_min_df, snr3_mid_df, snr3_max_df, offset=1)
        snr = pd.concat([snr1, snr2, snr3])
        snr = np.vstack([snr.index.values, snr[1].values])
        snr[0] *= 10
        return snr


class Sig2NoiseMUSE(Sig2NoiseVLT):

    """
    VLT/MUSE S/N Query (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr)

    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str band: Magnitude band. For valid options see s2n.vlt_options['src_target_mag_band (UVES)'].
    :param str magtype: Magnitude System. Either "Vega" or "AB"
    :param str template_type: Type of SED template. For now, only "template_spectrum" is supported.
    :param str template: Spectral template. For valid options see s2n.vlt_options['src_target_spec_type'].
    :param float redshift: Redshift of the target
    :param float airmass: Airmass of observation
    :param float moon_phase: Moon Phase between 0.0 (new) and 1.0 (full)
    :param str seeing: Seeing (FWHM) of observation in arcseconds.
        For valid options see s2n.vlt_options['sky_seeing'].
    :param str mode: MUSE instument mode. For valid options see s2n.vlt_options['muse_mode'].
    :param str spatial_binning: Spatial binning. For valid options see s2n.vlt_options['muse_spatial_binning'].
    :param str spectra_binning: Spectral binning. For valid options see s2n.vlt_options['muse_spectra_binning'].
    :param float target_offset: Displacement of source from fiber center.
    :param \**kwargs: Other entries in the ETC web form to set.
        To see what options are available, an inspection of the ETC website is necessary.
    """

    def __init__(
        self,
        exptime: float,
        mag: float,
        band: str = "V",
        magtype: str = "Vega",
        template_type: str = "template_spectrum",
        template: str = "Pickles_K2V",
        redshift: float = 0,
        airmass: float = 1.1,
        moon_phase: float = 0.0,
        seeing: str = "0.8",
        mode: str = "WFM_NONAO_N",
        spatial_binning: str = "3",
        spectra_binning: str = "1",
        target_offset: float = 0,
        **kwargs,
    ):
        Sig2NoiseVLT.__init__(
            self,
            "MUSE",
            exptime,
            mag,
            band,
            magtype,
            template_type,
            template,
            redshift,
            airmass,
            moon_phase,
            seeing,
            **kwargs,
        )
        self.url = "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr"
        if self.band not in vlt_options["src_target_mag_band (MUSE)"]:
            raise KeyError(
                f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band (MUSE)']}"
            )
        if mode not in vlt_options["muse_mode"]:
            raise KeyError(f"{mode} not one of {vlt_options['muse_mode']}")
        if spatial_binning not in vlt_options["muse_spatial_binning"]:
            raise KeyError(
                f"{spatial_binning} not one of {vlt_options['muse_spatial_binning']}"
            )
        if spectra_binning not in vlt_options["muse_spectra_binning"]:
            raise KeyError(
                f"{spectra_binning} not one of {vlt_options['muse_spectra_binning']}"
            )
        if not target_offset >= 0:
            raise ValueError("muse_target_offset must be positive")
        self.mode = mode
        self.spatial_binning = spatial_binning
        self.spectra_binning = spectra_binning
        self.target_offset = target_offset
        self.data = None

    def query_s2n(self):
        """
        Query the MUSE ETC (https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr)

        :return:
        """
        url = self.url
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form.new_control(type="select", name="SRC.TARGET.MAG.BAND", value="")
        form.new_control(type="select", name="SKY.SEEING.ZENITH.V", value="")
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.mag
        form["SRC.TARGET.MAG.BAND"] = self.band
        form["SRC.TARGET.MAG.SYSTEM"] = self.magtype
        form["SRC.TARGET.TYPE"] = self.template_type
        form["SRC.TARGET.SPEC.TYPE"] = self.template
        form["SRC.TARGET.REDSHIFT"] = self.redshift
        form["SRC.TARGET.GEOM"] = "seeing_ltd"
        # Sky Parameters
        form["SKY.AIRMASS"] = self.airmass
        form["SKY.MOON.FLI"] = self.moon_phase
        form["USR.SEEING.OR.IQ"] = "seeing_given"
        form["SKY.SEEING.ZENITH.V"] = self.seeing
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        # Instrument Specifics
        form["INS.NAME"] = "MUSE"
        form["INS.MODE"] = "swspectr"
        form["INS.MUSE.SETTING.KEY"] = self.mode
        form["INS.MUSE.SPATIAL.NPIX.LINEAR"] = self.spatial_binning
        form["INS.MUSE.SPECTRAL.NPIX.LINEAR"] = self.spectra_binning
        form["SRC.TARGET.GEOM.DISTANCE"] = self.target_offset
        form["USR.OBS.SETUP.TYPE"] = "givenexptime"
        form["DET.IR.NDIT"] = 1
        form["DET.IR.DIT"] = self.exptime
        form["USR.OUT.DISPLAY.SN.V.WAVELENGTH"] = 1
        for key in self.kwargs:
            form[key] = self.kwargs[key]
        self.data = browser.submit_selected()
        snr = self.parse_etc()
        return snr

    def parse_etc(self):
        snr_url = (
            "https://www.eso.org"
            + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        )
        snr_txt = requests.post(snr_url).text
        snr = pd.DataFrame([row.split(" ") for row in snr_txt.split("\n")[:-1]])
        snr.index = snr.pop(0)
        snr.sort_index(inplace=True)
        snr = np.vstack([snr.index.values, snr[1].values]).astype(float)
        snr[0] *= 10
        return snr


class Sig2NoiseMSE(Sig2NoiseQuery):
    """
    MSE S/N Query (http://etc-dev.cfht.hawaii.edu/mse/)

    :param float exptime: Exposure time in seconds
    :param float mag: Magnitude of source
    :param str template: Spectral template. For valid options see s2n.mse_options['template'].
    :param str spec_mode: MSE mode. Must be "LR" (low resolution), "MR" (medium resolution), or "HR" (high resolution)
    :param str band: Magnitude band. For valid options see s2n.mse_options['filter'].
    :param str airmass: Airmass of observation. Must be one of mse_options['airmass'].
    :param float seeing: Seeing (FWHM) of observation in arcseconds
    :param float skymag: Background sky magnitude of observation
    :param str src_type: Spatial profile of source. Must be one of mse_options['src_type'].
    :param float redshift: Redshift of the target.
    :param bool smoothed: If True, uses smoothed S/N,
    """

    def __init__(
        self,
        exptime: float,
        mag: float,
        template: str,
        spec_mode: str = "LR",
        band: str = "g",
        airmass: str = "1.2",
        seeing: float = 0.5,
        skymag: float = 20.7,
        src_type: str = "point",
        redshift: float = 0,
        smoothed: bool = False,
    ):
        Sig2NoiseQuery.__init__(self)
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
        if band not in mse_options["filter"]:
            raise KeyError(f"{band} not one of {mse_options['filter']}")
        if airmass not in mse_options["airmass"]:
            raise KeyError(f"{airmass} not one of {mse_options['airmass']}")
        if src_type not in mse_options["src_type"]:
            raise KeyError(f"{src_type} not one of {mse_options['src_type']}")
        self.exptime = exptime
        self.mag = mag
        self.template = template
        self.spec_mode = spec_mode
        self.band = band
        self.airmass = airmass
        self.seeing = seeing
        self.skymag = skymag
        self.src_type = src_type
        self.redshift = redshift
        self.smoothed = smoothed

    def query_s2n(self):
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
            + f"band={self.band}&"
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
        if self.smoothed:
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
        if self.spec_mode == "LR":
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
        elif self.spec_mode in ["MR", "HR"]:
            filler_x1 = np.linspace(x[0].max(), x[1].min(), 100)
            filler_x2 = np.linspace(x[1].max(), x[2].min(), 100)
            filler_y = np.zeros(100)
            x = np.concatenate([x[0], filler_x1, x[1], filler_x2, x[2]])
            y = np.concatenate([y[0], filler_y, y[1], filler_y, y[2]])
        else:
            raise RuntimeError(
                f"{self.spec_mode} not one of {mse_options['spec_mode']}"
            )
        snr = np.vstack([x, y])
        return snr


class Sig2NoiseLCO(Sig2NoiseQuery):
    """
    Superclass for LCO ETC Queries (http://alyth.lco.cl/gblanc_www/lcoetc/lcoetc_sspec.html)

    :param instrument: LCO instrument. Valid options are "MIKE", "LDSS3", "IMACS", and "MAGE".
    :param telescope: LCO telescope. "MAGELLAN1" for IMACS and MAGE. "MAGELLAN2" for LDSS3 and MIKE.
    :param exptime: Exposure time in seconds
    :param mag: Magnitude of source
    :param template: Spectral template. For valid options see s2n.lco_options['template'].
    :param band: Magnitude band. For valid options see s2n.lco_options['filter'].
    :param airmass: Airmass of observation
    :param seeing: Seeing (FWHM) of observation in arcseconds
    :param nmoon: Days from since new moon. For valid options see s2n.lco_options['nmoon'].
    :param nexp: Number of exposures
    :param slitwidth: Width of slit in arcseconds
    :param binspat: Binning in the spatial direction. For valid options see s2n.lco_options['binspat'].
    :param binspec: Binning in the spectral direction. For valid options see s2n.lco_options['binspec'].
    :param extract_ap: Size of extraction aperture in arcseconds.
    """
    def __init__(
        self,
        instrument: str,
        telescope: str,
        exptime: float,
        mag: float,
        template: str = "flat",
        band: str = "g",
        airmass: float = 1.1,
        seeing: float = 0.5,
        nmoon: str = "0",
        nexp: int = 1,
        slitwidth: float = 1.0,
        binspat: str = "3",
        binspec: str = "1",
        extract_ap: float = 1.5,
    ):
        Sig2NoiseQuery.__init__(self)
        self.url_base = "http://alyth.lco.cl/cgi-bin/gblanc_cgi/lcoetc/lcoetc_sspec.py"
        if template not in lco_options["template"]:
            raise KeyError(f"{template} not one of {lco_options['template']}")
        if band not in lco_options["tempfilter"]:
            raise KeyError(f"{band} not one of {lco_options['tempfilter']}")
        if telescope not in lco_options["telescope"]:
            raise KeyError(f"{telescope} not one of {lco_options['telescope']}")
        if instrument not in lco_options[telescope + "_instrument"]:
            raise KeyError(
                f"{instrument} not one of {lco_options[telescope+'_telescope']}"
            )
        if binspat not in lco_options["binspat"]:
            raise KeyError(f"{binspat} not one of {lco_options['binspat']}")
        if binspec not in lco_options["binspec"]:
            raise KeyError(f"{binspec} not one of {lco_options['binspec']}")
        if nmoon not in lco_options["nmoon"]:
            raise KeyError(f"{nmoon} not one of {lco_options['nmoon']}")
        if template == "flat":
            self.template = template
        else:
            self.template = f"{template}_Pickles.dat"
        self.abmag = mag
        self.tempfilter = f"sdss_{band}.dat"
        self.telescope = telescope
        self.instrument = instrument
        self.dslit = slitwidth
        self.binspat = binspat
        self.binspec = binspec
        self.nmoon = nmoon
        self.amass = airmass
        self.dpsf = seeing
        self.texp = exptime
        self.nexp = nexp
        self.aper = extract_ap

    def query_s2n(self):
        if not hasattr(self, "mode"):
            raise AttributeError(
                "Query has no attribute 'mode'."
                + "Try using the instrument specific query (e.g., Sig2NoiseMIKE) instead of this general one."
            )
        url = (
            f"{self.url_base}?"
            + f"template={self.template}&"
            + f"abmag={self.abmag}&"
            + f"tempfilter={self.tempfilter}&"
            + f"addline=0&"
            + f"linelam=5000&"
            + f"lineflux=1e-16&"
            + f"linefwhm=5.0&"
            + f"telescope={self.telescope}&"
            + f"instrument={self.instrument}&"
            + f"mode={self.mode}&"
            + f"dslit={self.dslit}&"
            + f"binspat={self.binspat}&"
            + f"binspec={self.binspec}&"
            + f"nmoon={self.nmoon}&"
            + f"amass={self.amass}&"
            + f"dpsf={self.dpsf}&"
            + f"texp={self.texp}&"
            + f"nexp={self.nexp}&"
            + f"aper={self.aper}&"
            + f"submitted=CALCULATE"
        )
        response = requests.post(url)
        data_url = response.text.split('href="')[1].split('" download>')[0]
        data_text = requests.post(data_url).text
        header = data_text.split("\n")[0]
        data = pd.DataFrame(
            [row.split(" ") for row in data_text.split("\n")[1:-1]],
            columns=header.split(" ")[1:],
        )
        snr = np.vstack(
            [data["Wavelength_[A]"].values, data["S/N_Aperture_Coadd"]]
        ).astype(float)
        return snr


class Sig2NoiseIMACS(Sig2NoiseLCO):
    """
    Magellan/IMACS S/N Query (http://alyth.lco.cl/gblanc_www/lcoetc/lcoetc_sspec.html)

    :param mode: IMACS mode. For valid options see s2n.lco_options['IMACS_mode'].
    :param exptime: Exposure time in seconds
    :param mag: Magnitude of source
    :param template: Spectral template. For valid options see s2n.lco_options['template'].
    :param band: Magnitude band. For valid options see s2n.lco_options['filter'].
    :param airmass: Airmass of observation
    :param seeing: Seeing (FWHM) of observation in arcseconds
    :param nmoon: Days from since new moon. For valid options see s2n.lco_options['nmoon'].
    :param nexp: Number of exposures
    :param slitwidth: Width of slit in arcseconds
    :param binspat: Binning in the spatial direction. For valid options see s2n.lco_options['binspat'].
    :param binspec: Binning in the spectral direction. For valid options see s2n.lco_options['binspec'].
    :param extract_ap: Size of extraction aperture in arcseconds.
    """
    def __init__(
        self,
        mode: str,
        exptime: float,
        mag: float,
        template: str = "flat",
        band: str = "g",
        airmass: float = 1.1,
        seeing: float = 0.5,
        nmoon: str = "0",
        nexp: int = 1,
        slitwidth: float = 1.0,
        binspat: str = "3",
        binspec: str = "1",
        extract_ap: float = 1.5,
    ):
        Sig2NoiseLCO.__init__(
            self,
            "IMACS",
            "MAGELLAN1",
            exptime,
            mag,
            template,
            band,
            airmass,
            seeing,
            nmoon,
            nexp,
            slitwidth,
            binspat,
            binspec,
            extract_ap,
        )
        if mode not in lco_options["IMACS_mode"]:
            raise KeyError(f"{mode} not one of {lco_options['IMACS_mode']}")
        self.mode = mode


class Sig2NoiseMAGE(Sig2NoiseLCO):
    """
    Magellan/MAGE S/N Query (http://alyth.lco.cl/gblanc_www/lcoetc/lcoetc_sspec.html)

    :param mode: MAGE mode. "ECHELLETTE" is currently the only option.
    :param exptime: Exposure time in seconds
    :param mag: Magnitude of source
    :param template: Spectral template. For valid options see s2n.lco_options['template'].
    :param band: Magnitude band. For valid options see s2n.lco_options['filter'].
    :param airmass: Airmass of observation
    :param seeing: Seeing (FWHM) of observation in arcseconds
    :param nmoon: Days from since new moon. For valid options see s2n.lco_options['nmoon'].
    :param nexp: Number of exposures
    :param slitwidth: Width of slit in arcseconds
    :param binspat: Binning in the spatial direction. For valid options see s2n.lco_options['binspat'].
    :param binspec: Binning in the spectral direction. For valid options see s2n.lco_options['binspec'].
    :param extract_ap: Size of extraction aperture in arcseconds.
    """
    def __init__(
        self,
        mode: str,
        exptime: float,
        mag: float,
        template: str = "flat",
        band: str = "g",
        airmass: float = 1.1,
        seeing: float = 0.5,
        nmoon: str = "0",
        nexp: int = 1,
        slitwidth: float = 1.0,
        binspat: str = "3",
        binspec: str = "1",
        extract_ap: float = 1.5,
    ):
        Sig2NoiseLCO.__init__(
            self,
            "MAGE",
            "MAGELLAN1",
            exptime,
            mag,
            template,
            band,
            airmass,
            seeing,
            nmoon,
            nexp,
            slitwidth,
            binspat,
            binspec,
            extract_ap,
        )
        if mode not in lco_options["MAGE_mode"]:
            raise KeyError(f"{mode} not one of {lco_options['MAGE_mode']}")
        self.mode = mode


class Sig2NoiseMIKE(Sig2NoiseLCO):
    """
    Magellan/MIKE S/N Query (http://alyth.lco.cl/gblanc_www/lcoetc/lcoetc_sspec.html)

    :param mode: MIKE mode. Valid options are "BLUE" and "RED".
    :param exptime: Exposure time in seconds
    :param mag: Magnitude of source
    :param template: Spectral template. For valid options see s2n.lco_options['template'].
    :param band: Magnitude band. For valid options see s2n.lco_options['filter'].
    :param airmass: Airmass of observation
    :param seeing: Seeing (FWHM) of observation in arcseconds
    :param nmoon: Days from since new moon. For valid options see s2n.lco_options['nmoon'].
    :param nexp: Number of exposures
    :param slitwidth: Width of slit in arcseconds
    :param binspat: Binning in the spatial direction. For valid options see s2n.lco_options['binspat'].
    :param binspec: Binning in the spectral direction. For valid options see s2n.lco_options['binspec'].
    :param extract_ap: Size of extraction aperture in arcseconds.
    """
    def __init__(
        self,
        mode: str,
        exptime: float,
        mag: float,
        template: str = "flat",
        band: str = "g",
        airmass: float = 1.1,
        seeing: float = 0.5,
        nmoon: str = "0",
        nexp: int = 1,
        slitwidth: float = 1.0,
        binspat: str = "3",
        binspec: str = "1",
        extract_ap: float = 1.5,
    ):
        Sig2NoiseLCO.__init__(
            self,
            "MIKE",
            "MAGELLAN2",
            exptime,
            mag,
            template,
            band,
            airmass,
            seeing,
            nmoon,
            nexp,
            slitwidth,
            binspat,
            binspec,
            extract_ap,
        )
        if mode not in lco_options["MIKE_mode"]:
            raise KeyError(f"{mode} not one of {lco_options['MIKE_mode']}")
        self.mode = mode


class Sig2NoiseLDSS3(Sig2NoiseLCO):
    """
    Magellan/LDSS-3 S/N Query (http://alyth.lco.cl/gblanc_www/lcoetc/lcoetc_sspec.html)

    :param mode: LDSS-3 mode. Valid options are "VPHALL", "VPHBLUE", and "VPHRED".
    :param exptime: Exposure time in seconds
    :param mag: Magnitude of source
    :param template: Spectral template. For valid options see s2n.lco_options['template'].
    :param band: Magnitude band. For valid options see s2n.lco_options['filter'].
    :param airmass: Airmass of observation
    :param seeing: Seeing (FWHM) of observation in arcseconds
    :param nmoon: Days from since new moon. For valid options see s2n.lco_options['nmoon'].
    :param nexp: Number of exposures
    :param slitwidth: Width of slit in arcseconds
    :param binspat: Binning in the spatial direction. For valid options see s2n.lco_options['binspat'].
    :param binspec: Binning in the spectral direction. For valid options see s2n.lco_options['binspec'].
    :param extract_ap: Size of extraction aperture in arcseconds.
    """
    def __init__(
        self,
        mode: str,
        exptime: float,
        mag: float,
        template: str = "flat",
        band: str = "g",
        airmass: float = 1.1,
        seeing: float = 0.5,
        nmoon: str = "0",
        nexp: int = 1,
        slitwidth: float = 1.0,
        binspat: str = "3",
        binspec: str = "1",
        extract_ap: float = 1.5,
    ):
        Sig2NoiseLCO.__init__(
            self,
            "LDSS3",
            "MAGELLAN2",
            exptime,
            mag,
            template,
            band,
            airmass,
            seeing,
            nmoon,
            nexp,
            slitwidth,
            binspat,
            binspec,
            extract_ap,
        )
        if mode not in lco_options["LDSS3_mode"]:
            raise KeyError(f"{mode} not one of {lco_options['LDSS3_mode']}")
        self.mode = mode


def calculate_mods_snr(
    F: np.ndarray,
    wave: np.ndarray,
    t_exp: float,
    airmass: float = 1.1,
    slitloss: float = 0.76,
    mode: str = "dichroic",
    side: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate S/N for LBT/MODS. Based on the calculations and data presented here:
    https://sites.google.com/a/lbto.org/mods/preparing-to-observe/sensitivity

    :param np.ndarray F: Flux (ergs s^-1 Angstrom^-1  cm^-2)
    :param np.ndarray wave: Wavelength array (Angstrom)
    :param float t_exp: Exposure time in seconds
    :param float airmass: Airmass of observation
    :param float slitloss: Slit loss factor (i.e., the fraction of the flux that makes it through the slit).
    :param str mode: "dichroic" for both red and blue detectors or "direct" for just one or the other.
    :param Optional[str] side: Detector to use if mode="direct". Must be either "red" or "blue".
    :return np.ndarray: S/N as a function of wavelength for LBT/MODS
    """
    if mode not in ["dichroic", "direct"]:
        raise KeyError("mode must be either 'dichroic' or 'direct'.")
    if mode == "direct" and side not in ["red", "blue"]:
        raise KeyError("side must be either 'red' or 'blue' if mode is 'direct'.")
    if len(F) != len(wave):
        raise ValueError("Flux and wavelength must be the same length.")
    if airmass < 1.0:
        raise ValueError("Airmass must be greater than or equal to 1.")
    log_t_exp = np.log10(t_exp)
    log_F = np.log10(F)
    g_blue = 2.5  # electron / ADU
    g_red = 2.6  # electron / ADU
    sigma_RO_red = 2.5  # electron
    sigma_RO_blue = 2.5  # electron
    A_per_pix_red = 0.85
    A_per_pix_blue = 0.50
    slitloss = 0.76
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
        S_red = 10 ** log_S_red * slitloss * A_per_pix_red
        S_blue = 10 ** log_S_blue * slitloss * A_per_pix_blue
        snr_red = g_red * S_red / np.sqrt(g_red * S_red + sigma_RO_red ** 2)
        snr_blue = g_blue * S_blue / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
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
            S_red = 10 ** log_S_red * slitloss * A_per_pix_red
            snr = g_red * S_red / np.sqrt(g_red * S_red + sigma_RO_red ** 2)
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
            S_blue = 10 ** log_S_blue * slitloss * A_per_pix_blue
            snr = g_blue * S_blue / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
        else:
            raise RuntimeError("Improper side argument")
    else:
        raise RuntimeError("Improper mode argument")
    return np.array([wave, snr])


def calculate_fobos_snr(
    spec_file: Optional[str] = None,
    spec_wave: Union[str, float] = "WAVE",
    spec_wave_units: str = "angstrom",
    spec_flux: Union[str, float] = "FLUX",
    spec_flux_units: Optional[str] = None,
    spot_fwhm: float = 5.8,
    spec_res_indx: Optional[Union[str, float]] = None,
    spec_res_value: Optional[float] = None,
    spec_table: Optional[Union[str, float]] = None,
    mag: float = 24.0,
    mag_band: str = "g",
    mag_system: str = "AB",
    sky_mag: Optional[float] = None,
    sky_mag_band: str = "g",
    sky_mag_system: str = "AB",
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
) -> np.ndarray:
    """
    This is slightly modified code from https://github.com/Keck-FOBOS/enyo/blob/master/python/enyo/scripts/fobos_etc.py

    :param Optional[str] spec_file: A fits or ascii file with the object spectrum to use. If None, a flat spectrum is used.
    :param Union[str,float]spec_wave: Extension or column number with the wavelengths.
    :param str spec_wave_units: Wavelength units
    :param Union[str,float] spec_flux: Extension or column number with the flux.
    :param Optional[str] spec_flux_units: Input units of the flux density. Must be interpretable by astropy.units.Unit.
        Assumes 1e-17 erg / (cm2 s angstrom) if units are not provided.
    :param float spot_fwhm: FHWM of the monochromatic spot size on the detector in pixels.
    :param Optional[Union[str,float]] spec_res_indx: Extension or column number with the flux.
    :param Optional[float] spec_res_value: Single value for the spectral resolution (R = lambda/dlambda) for the full spectrum.
    :param Optional[Union[str,float]] spec_table: Extension in the fits file with the binary table data.
    :param float mag: Total apparent magnitude of the source
    :param str mag_band: Broad-band used for the provided magnitude. Must be u, g, r, i, or z.
    :param str mag_system: Magnitude system. Must be either AB or Vega.
    :param Optional[float] sky_mag: Surface brightness of the sky in mag/arcsec^2 in the defined broadband.
        If not provided, default dark-sky spectrum is used.
    :param str sky_mag_band: Broad-band used for the provided sky surface brightness. Must be u, g, r, i, or z.
    :param str sky_mag_system: Magnitude system. Must be either AB or Vega.
    :param float redshift: Redshift of the object, z
    :param Optional[str] emline: File with emission lines to add to the spectrum.
    :param Optional[Tuple[float,float,float,float]] sersic: Use a Sersic profile to describe the object surface-brightness  distribution; order
        must be effective radius, Sersic index, ellipticity (1-b/a), position angle (deg).
    :param bool uniform: Instead of a point source or Sersic profile,
        assume the surface brightness distribution is uniform over the fiber face.
        If set, the provided magnitude is assumed to be a surface brightness.
        See the MAG option.
    :param float exptime: Exposure time (s)
    :param float fwhm: On-sky PSF FWHM (arcsec)
    :param float airmass: Airmass
    :param str snr_units: The units for the S/N. Options are pixel, angstrom, resolution.
    :param float sky_err: The fraction of the Poisson error in the sky incurred when subtracting the sky from the observation.
        Set to 0 for a sky subtraction that adds no error to the sky-subtracted spectrum;
        set to 1 for a sky-subtraction error that is the same as the Poisson error in the sky spectrum
        acquired during the observation.
    :param bool print_summary: If True, prints a summary of the calculations.
    :return np.ndarray: S/N as a function of wavelength for Keck/FOBOS
    """
    try:
        from enyo.etc import (
            spectrum,
            efficiency,
            telescopes,
            aperture,
            detector,
            extract,
        )
        from enyo.etc.observe import Observation
        from enyo.scripts.fobos_etc import (
            get_wavelength_vector,
            read_emission_line_database,
            get_spectrum,
            get_sky_spectrum,
            get_source_distribution,
        )
    except ImportError:
        raise ImportError(
            "To calculate FOBOS S/N you must first install the FOBOS ETC.\n "
            + "See <> for installation instructions."
        )
    if sky_err < 0 or sky_err > 1:
        raise ValueError("--sky_err option must provide a value between 0 and 1.")
    # Constants:
    resolution = 3500.0  # lambda/dlambda
    fiber_diameter = 0.8  # Arcsec
    rn = 2.0  # Detector readnoise (e-)
    dark = 0.0  # Detector dark-current (e-/s)
    # Temporary numbers that assume a given spectrograph PSF and LSF.
    # Assume 3 pixels per spectral and spatial FWHM.
    spatial_fwhm = spot_fwhm
    spectral_fwhm = spot_fwhm
    # Get source spectrum in 1e-17 erg/s/cm^2/angstrom. Currently, the
    # source spectrum is assumed to be
    #   - normalized by the total integral of the source flux
    #   - independent of position within the source
    dw = 1 / spectral_fwhm / resolution / np.log(10)
    wavelengths = [3100, 10000, dw]
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
    sky_spectrum = get_sky_spectrum(
        sky_mag, mag_band=sky_mag_band, mag_system=sky_mag_system
    )
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
    # Get the spectrograph throughput (circa June 2018; needs to
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
    snr_label = "S/N per {0}".format(
        "R element" if snr_units == "resolution" else snr_units
    )
    if print_summary:
        # Report
        g = efficiency.FilterResponse(band="g")
        r = efficiency.FilterResponse(band="r")
        iband = efficiency.FilterResponse(band="i")
        print("-" * 70)
        print("{0:^70}".format("FOBOS S/N Calculation (v0.2)"))
        print("-" * 70)
        print("Compute time: {0} seconds".format(time.perf_counter() - t))
        print(
            "Object g- and r-band AB magnitude: {0:.1f} {1:.1f}".format(
                spec.magnitude(band=g), spec.magnitude(band=r)
            )
        )
        print(
            "Sky g- and r-band AB surface brightness: {0:.1f} {1:.1f}".format(
                sky_spectrum.magnitude(band=g), sky_spectrum.magnitude(band=r)
            )
        )
        print("Exposure time: {0:.1f} (s)".format(exptime))
        if not uniform:
            print("Aperture Loss: {0:.1f}%".format((1 - obs.aperture_factor) * 100))
        print(
            "Extraction Loss: {0:.1f}%".format(
                (1 - obs.extraction.spatial_efficiency) * 100
            )
        )
        print("Median {0}: {1:.1f}".format(snr_label, np.median(snr.flux)))
        print(
            "g-band weighted mean {0} {1:.1f}".format(
                snr_label, np.sum(g(snr.wave) * snr.flux) / np.sum(g(snr.wave))
            )
        )
        print(
            "r-band weighted mean {0} {1:.1f}".format(
                snr_label, np.sum(r(snr.wave) * snr.flux) / np.sum(r(snr.wave))
            )
        )
        print(
            "i-band weighted mean {0} {1:.1f}".format(
                snr_label, np.sum(iband(snr.wave) * snr.flux) / np.sum(iband(snr.wave))
            )
        )
    return np.vstack([snr.wave, snr.flux])


def calculate_wfos_snr(
    spec_file: Optional[str] = None,
    spec_wave: Union[str, float] = "WAVE",
    spec_wave_units: str = "angstrom",
    spec_flux: Union[str, float] = "FLUX",
    spec_flux_units: Optional[str] = None,
    #spot_fwhm: float = 5.8,  #  Not used
    spec_res_indx: Optional[Union[str, float]] = None,
    spec_res_value: Optional[float] = None,
    spec_table: Optional[Union[str, float]] = None,
    mag: float = 24.0,
    mag_band: str = "g",
    mag_system: str = "AB",
    #sky_mag: Optional[float] = None,  #  Not used
    #sky_mag_band: str = "g",  #  Not used
    #sky_mag_system: str = "AB",  #  Not used
    redshift: float = 0.0,
    emline: Optional[str] = None,
    sersic: Optional[Tuple[float, float, float, float]] = None,
    uniform: bool = False,
    exptime: float = 3600.0,
    fwhm: float = 0.65,
    airmass: float = 1.0,
    snr_units: str = "pixel",
    sky_err: float = 0.1,
    # WFOS specifics
    refl: str = "req",
    blue_grat: str = "B1210",
    blue_wave: Optional[float] = None,
    blue_angle: Optional[float] = None,
    blue_binning: Optional[Tuple[int, int]] = (1, 1),
    red_grat: str = "R680",
    red_wave: Optional[float] = None,
    red_angle: Optional[float] = None,
    red_binning: Optional[Tuple[int, int]] = (1, 1),
    slit: Optional[Tuple[float, float, float, float, float]] = (
        0.0,
        0.0,
        0.75,
        5.0,
        0.0,
    ),
    extract_size: Optional[float] = None,
    return_R: bool = False,
    print_summary: bool = True,
)  -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    This is slightly modified code from https://github.com/Keck-FOBOS/enyo/blob/master/enyo/scripts/wfos_etc.py

    :param Optional[str] spec_file: A fits or ascii file with the object spectrum to use. If None, a flat spectrum is used.
    :param Union[str,float]spec_wave: Extension or column number with the wavelengths.
    :param str spec_wave_units: Wavelength units
    :param Union[str,float] spec_flux: Extension or column number with the flux.
    :param Optional[str] spec_flux_units: Input units of the flux density. Must be interpretable by astropy.units.Unit.
        Assumes 1e-17 erg / (cm2 s angstrom) if units are not provided.
    :param Optional[Union[str,float]] spec_res_indx: Extension or column number with the flux.
    :param Optional[float] spec_res_value: Single value for the spectral resolution (R = lambda/dlambda) for the full spectrum.
    :param Optional[Union[str,float]] spec_table: Extension in the fits file with the binary table data.
    :param float mag: Total apparent magnitude of the source
    :param str mag_band: Broad-band used for the provided magnitude. Must be u, g, r, i, or z.
    :param str mag_system: Magnitude system. Must be either AB or Vega.
    :param float redshift: Redshift of the object, z
    :param Optional[str] emline: File with emission lines to add to the spectrum.
    :param Optional[Tuple[float,float,float,float]] sersic: Use a Sersic profile to describe the object surface-brightness  distribution; order
        must be effective radius, Sersic index, ellipticity (1-b/a), position angle (deg).
    :param bool uniform: Instead of a point source or Sersic profile,
        assume the surface brightness distribution is uniform over the fiber face.
        If set, the provided magnitude is assumed to be a surface brightness.
        See the MAG option.
    :param float exptime: Exposure time (s)
    :param float fwhm: On-sky PSF FWHM (arcsec)
    :param float airmass: Airmass
    :param str snr_units: The units for the S/N. Options are pixel, angstrom, resolution.
    :param float sky_err: The fraction of the Poisson error in the sky incurred when subtracting the sky from the observation.
        Set to 0 for a sky subtraction that adds no error to the sky-subtracted spectrum;
        set to 1 for a sky-subtraction error that is the same as the Poisson error in the sky spectrum
        acquired during the observation.
    :param str refl: Select the reflectivity curve for TMT.
        Must be either 'req' or 'goal' for the required or goal reflectivity performance.
    :param str blue_grat: Grating to use in the blue arm.
        For valid options see enyo.etc.spectrographs.WFOSGrating.available_gratings.keys()
    :param Optional[float] blue_wave: Central wavelength for the blue arm.
        If None, will use the peak-efficiency wavelength.
    :param Optional[float] blue_angle: Grating angle for blue grating.
        If None, will use then angle the provides the best efficiency for the on-axis spectrum.
    :param Optional[Tuple[int,int]] blue_binning: On-chip binning for the blue grating. Order is spectral then spatial.
        I.e., to bin 2 pixels spectrally and no binning spatial, use (2, 1)
    :param str red_grat: Grating to use in the red arm.
        For valid options see enyo.etc.spectrographs.WFOSGrating.available_gratings.keys()
    :param Optional[float] red_wave: Central wavelength for the red arm.
        If None, will use the peak-efficiency wavelength.
    :param Optional[float] red_angle: Grating angle for red grating.
        If None, will use then angle the provides the best efficiency for the on-axis spectrum.
    :param Optional[Tuple[int,int]] red_binning: On-chip binning for the red grating. Order is spectral then spatial.
        I.e., to bin 2 pixels spectrally and no binning spatial, use (2, 1)
    :param Optional[Tuple[float,float,float,float,float]] slit: Slit properties:
        x field center, y field center, width, length, rotation.
        The rotation is in degrees, everything else is in on-sky arcsec.
        The slit width is in the *unrotated* frame, meaning the effective slit width for a rotated slit is
        slit_width/cos(rotation). For the field center, x is along the dispersion direction with a valid range of
        +/- 90 arcsec, and y is in the cross-dispersion direction with a valid range of +/- 249 arcsec.
        Coordinate (0,0) is on axis.
    :param Optional[float] extract_size: Extraction aperture in arcsec *along the slit* centered on the source.
        At the detector, the extraction aperture is narrower by cos(slit rotation).
        If not provided, set to the FWHM of the seeing disk.
    :param bool return_R: If True, also returns the resolution as a function of wavelength.
    :param bool print_summary: If True, prints a summary of the calculations.
    :return Union[np.ndarray,Tuple[np.ndarray,np.ndarray]]: S/N as a function of wavelength for Keck/WFOS.
        If return_R, a tuple of S/N and resolving power as a function of wavelength.
    """
    try:
        from enyo.etc import (
            spectrum,
            efficiency,
            telescopes,
            aperture,
            detector,
            extract,
        )
        from enyo.etc.observe import Observation
        from enyo.etc.spectrographs import TMTWFOSBlue, TMTWFOSRed, WFOSGrating
        from enyo.scripts.wfos_etc import (
            get_source_distribution,
            get_wavelength_vector,
            observed_spectrum,
            read_emission_line_database,
            read_spectrum,
        )
    except ImportError:
        raise ImportError(
            "To calculate WFOS S/N you must first install the WFOS ETC.\n "
            + "See <> for installation instructions."
        )
    if sky_err < 0 or sky_err > 1:
        raise ValueError("--sky_err option must provide a value between 0 and 1.")
    # Extract the slit properties for clarity
    slit_x, slit_y, slit_width, slit_length, slit_rotation = slit
    effective_slit_width = slit_width / np.cos(np.radians(slit_rotation))
    _extract_length = fwhm if extract_size is None else extract_size
    # Slit aperture. This representation of the slit is *always*
    # centered at (0,0). Set the aperture based on the extraction
    # length for now.
    slit = aperture.SlitAperture(
        0.0, 0.0, slit_width, _extract_length, rotation=slit_rotation
    )
    # Get the source distribution.  If the source is uniform, onsky is None.
    onsky = get_source_distribution(fwhm, uniform, sersic)
    # Sky spectrum and atmospheric throughput
    sky_spectrum = spectrum.MaunakeaSkySpectrum()
    atmospheric_throughput = efficiency.AtmosphericThroughput(airmass=airmass)
    # Emission lines to add
    emline_db = None if emline is None else read_emission_line_database(emline)
    # Setup the raw object spectrum
    if spec_file is None:
        wavelengths = [3100, 10000, 1e-5]
        wave = get_wavelength_vector(*wavelengths)
        obj_spectrum = spectrum.ABReferenceSpectrum(wave, log=True)
    else:
        obj_spectrum = read_spectrum(
            spec_file,
            spec_wave,
            spec_wave_units,
            spec_flux,
            spec_flux_units,
            spec_res_indx,
            spec_res_value,
            spec_table,
        )
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Setup the instrument arms
    # -------------------------------------------------------------------
    # Blue Arm
    blue_arm = TMTWFOSBlue(
        reflectivity=refl,
        grating=blue_grat,
        cen_wave=blue_wave,
        grating_angle=blue_angle,
    )
    # Pixels per resolution element
    blue_res_pix = (
        blue_arm.resolution_element(slit_width=effective_slit_width, units="pixels")
        / blue_binning[0]
    )
    # Get the wavelength range for each arm
    blue_wave_lim = blue_arm.wavelength_limits(slit_x, slit_y, add_grating_limits=True)
    # Setup dummy wavelength vectors to get something appropriate for sampling
    max_resolution = blue_arm.resolution(
        blue_wave_lim[1], x=slit_x, slit_width=effective_slit_width
    )
    # Set the wavelength vector to allow for a regular, logarithmic binning
    dw = 1 / blue_res_pix / max_resolution / np.log(10)
    blue_wave = get_wavelength_vector(blue_wave_lim[0], blue_wave_lim[1], dw)
    resolution = blue_arm.resolution(
        blue_wave, x=slit_x, slit_width=effective_slit_width
    )
    blue_spec = observed_spectrum(
        obj_spectrum,
        blue_wave,
        resolution,
        mag=mag,
        mag_band=mag_band,
        mag_system=mag_system,
        redshift=redshift,
        emline_db=emline_db,
    )
    blue_R_interp = interp1d(
        blue_wave,
        resolution,
        fill_value=(resolution[0], resolution[-1]),
        bounds_error=False,
    )
    # Resample to linear to better match what's expected for the detector
    blue_ang_per_pix = (
        blue_arm.resolution_element(
            wave=blue_wave_lim, slit_width=effective_slit_width, units="angstrom"
        )
        / blue_res_pix
    )
    blue_wave = get_wavelength_vector(
        blue_wave_lim[0], blue_wave_lim[1], np.mean(blue_ang_per_pix), linear=True
    )
    blue_spec = blue_spec.resample(wave=blue_wave, log=False)
    # Spectrograph arm efficiency (this doesn't include the telescope)
    blue_arm_eff = blue_arm.efficiency(
        blue_spec.wave, x=slit_x, y=slit_y, same_type=False
    )
    # System efficiency combines the spectrograph and the telescope
    blue_thru = efficiency.SystemThroughput(
        wave=blue_spec.wave,
        spectrograph=blue_arm_eff,
        telescope=blue_arm.telescope.throughput,
    )
    # Extraction: makes simple assumptions about the monochromatic
    # image and extracts the flux within the aperture, assuming the
    # flux from both the object and sky is uniformly distributed across
    # all detector pixels (incorrect!).
    # Extraction width in pixels
    spatial_width = (
        slit.length
        * np.cos(np.radians(slit.rotation))
        / blue_arm.pixelscale
        / blue_binning[1]
    )
    blue_ext = extract.Extraction(
        blue_arm.det, spatial_width=spatial_width, profile="uniform"
    )
    # Perform the observation
    blue_obs = Observation(
        blue_arm.telescope,
        sky_spectrum,
        slit,
        exptime,
        blue_arm.det,
        system_throughput=blue_thru,
        atmospheric_throughput=atmospheric_throughput,
        airmass=airmass,
        onsky_source_distribution=onsky,
        source_spectrum=blue_spec,
        extraction=blue_ext,
        snr_units=snr_units,
    )
    # Construct the S/N spectrum
    blue_snr = blue_obs.snr(sky_sub=True, sky_err=sky_err)
    blue_R = blue_R_interp(blue_snr.wave)
    # -------------------------------------------------------------------
    # Red Arm
    red_arm = TMTWFOSRed(
        reflectivity=refl, grating=red_grat, cen_wave=red_wave, grating_angle=red_angle
    )
    # Pixels per resolution element
    red_res_pix = (
        red_arm.resolution_element(slit_width=effective_slit_width, units="pixels")
        / red_binning[0]
    )
    # Get the wavelength range for each arm
    red_wave_lim = red_arm.wavelength_limits(slit_x, slit_y, add_grating_limits=True)
    # Setup dummy wavelength vectors to get something appropriate for sampling
    max_resolution = red_arm.resolution(
        red_wave_lim[1], x=slit_x, slit_width=effective_slit_width
    )
    # Set the wavelength vector to allow for a regular, logarithmic binning
    dw = 1 / red_res_pix / max_resolution / np.log(10)
    red_wave = get_wavelength_vector(red_wave_lim[0], red_wave_lim[1], dw)
    resolution = red_arm.resolution(red_wave, x=slit_x, slit_width=effective_slit_width)
    red_spec = observed_spectrum(
        obj_spectrum,
        red_wave,
        resolution,
        mag=mag,
        mag_band=mag_band,
        mag_system=mag_system,
        redshift=redshift,
        emline_db=emline_db,
    )
    # Resample to linear to better match what's expected for the detector
    red_ang_per_pix = (
        red_arm.resolution_element(
            wave=red_wave_lim, slit_width=effective_slit_width, units="angstrom"
        )
        / red_res_pix
    )
    red_wave = get_wavelength_vector(
        red_wave_lim[0], red_wave_lim[1], np.mean(red_ang_per_pix), linear=True
    )
    ree_spec = red_spec.resample(wave=red_wave, log=False)
    # Spectrograph arm efficiency (this doesn't include the telescope)
    red_arm_eff = red_arm.efficiency(red_spec.wave, x=slit_x, y=slit_y, same_type=False)
    # System efficiency combines the spectrograph and the telescope
    red_thru = efficiency.SystemThroughput(
        wave=red_spec.wave,
        spectrograph=red_arm_eff,
        telescope=red_arm.telescope.throughput,
    )
    # Extraction: makes simple assumptions about the monochromatic
    # image and extracts the flux within the aperture, assuming the
    # flux from both the object and sky is uniformly distributed across
    # all detector pixels (incorrect!).
    # Extraction width in pixels
    spatial_width = (
        slit.length
        * np.cos(np.radians(slit.rotation))
        / red_arm.pixelscale
        / red_binning[1]
    )
    red_ext = extract.Extraction(
        red_arm.det, spatial_width=spatial_width, profile="uniform"
    )
    # Perform the observation
    red_obs = Observation(
        red_arm.telescope,
        sky_spectrum,
        slit,
        exptime,
        red_arm.det,
        system_throughput=red_thru,
        atmospheric_throughput=atmospheric_throughput,
        airmass=airmass,
        onsky_source_distribution=onsky,
        source_spectrum=red_spec,
        extraction=red_ext,
        snr_units=snr_units,
    )
    # Construct the S/N spectrum
    red_snr = red_obs.snr(sky_sub=True, sky_err=sky_err)
    # Set the wavelength vector
    dw = 1 / (5 if red_res_pix > 5 else red_res_pix) / max_resolution / np.log(10)
    red_wave = get_wavelength_vector(red_wave_lim[0], red_wave_lim[1], dw)
    resolution = red_arm.resolution(red_wave, x=slit_x, slit_width=effective_slit_width)
    red_spec = observed_spectrum(
        obj_spectrum,
        red_wave,
        resolution,
        mag=mag,
        mag_band=mag_band,
        mag_system=mag_system,
        redshift=redshift,
        emline_db=emline_db,
    )
    red_R = interp1d(
        red_wave,
        resolution,
        fill_value=(resolution[0], resolution[-1]),
        bounds_error=False,
    )(red_snr.wave)
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    snr_label = "S/N per {0}".format(
        "R element" if snr_units == "resolution" else snr_units
    )
    if print_summary:
        g = efficiency.FilterResponse(band="g")
        r = efficiency.FilterResponse(band="r")
        print("-" * 70)
        print("{0:^70}".format("WFOS S/N Calculation (v0.1)"))
        print("-" * 70)
        print(
            "Object g- and r-band AB magnitude: {0:.1f} {1:.1f}".format(
                obj_spectrum.magnitude(band=g), obj_spectrum.magnitude(band=r)
            )
        )
        print(
            "Sky g- and r-band AB surface brightness: {0:.1f} {1:.1f}".format(
                sky_spectrum.magnitude(band=g), sky_spectrum.magnitude(band=r)
            )
        )
        print("Exposure time: {0:.1f} (s)".format(exptime))
        if not uniform:
            print("Aperture Loss: {0:.1f}%".format((1 - red_obs.aperture_factor) * 100))

    if blue_snr.wave.max() > red_snr.wave.min():
        bwave_overlap = blue_snr.wave[blue_snr.wave > red_snr.wave.min()]
        rwave_overlap = red_snr.wave[red_snr.wave < blue_snr.wave.max()]
        bsnr_overlap = blue_snr.flux[blue_snr.wave > red_snr.wave.min()]
        rsnr_overlap = red_snr.flux[red_snr.wave < blue_snr.wave.max()]
        diff = np.sqrt(
            (bwave_overlap[:, np.newaxis] - rwave_overlap[np.newaxis, :]) ** 2
            + (bsnr_overlap[:, np.newaxis] - rsnr_overlap[np.newaxis, :]) ** 2
        )
        bintersect_ind, rintersect_ind = np.unravel_index(
            np.argmin(diff, axis=None), diff.shape
        )
        i = 0
        while bwave_overlap[bintersect_ind] > rwave_overlap[rintersect_ind + i]:
            i += 1
        bind = find_nearest_idx(blue_snr.wave, bwave_overlap[bintersect_ind])
        rind = find_nearest_idx(red_snr.wave, rwave_overlap[rintersect_ind + i])
    else:
        bind = len(blue_snr.wave)
        rind = 0
    snr = np.vstack(
        [
            np.concatenate([blue_snr.wave[:bind], red_snr.wave[rind:]]),
            np.concatenate([blue_snr.flux[:bind], red_snr.flux[rind:]]),
        ]
    )
    resolution = np.vstack(
        [
            np.concatenate([blue_snr.wave[:bind], red_snr.wave[rind:]]),
            np.concatenate([blue_R[:bind], red_R[rind:]]),
        ]
    )
    if return_R:
        return snr, resolution
    else:
        return snr


def calculate_muse_snr(
    wave,
    flux,
    exptime,
    nexp,
    blueMUSE=False,
    airmass=1.0,
    seeing=0.8,
    moon="d",
    pointsource=True,
    nspatial=3,
    nspectral=1,
):
    """
    This code is adapted from https://git-cral.univ-lyon1.fr/johan.richard/BlueMUSE-ETC

    :param wave:
    :param flux:
    :param exptime:
    :param nexp:
    :param blueMUSE:
    :param airmass:
    :param seeing:
    :param moon:
    :param pointsource:
    :param nspatial:
    :param nspectral:
    :return:
    """
    MUSE_etc_dir = etc_file_dir.joinpath("MUSE")
    muse_files = [
        MUSE_etc_dir.joinpath("NewBlueMUSE_noatm.txt"),
        MUSE_etc_dir.joinpath("radiance_airmass1.0_0.5moon.txt"),
        MUSE_etc_dir.joinpath("radiance_airmass1.0_newmoon.txt"),
        MUSE_etc_dir.joinpath("transmission_airmass1.txt"),
        MUSE_etc_dir.joinpath("WFM_NONAO_N.dat.txt"),
    ]
    if not all([file.exists() for file in muse_files]):
        download_bluemuse_files()
    ron = 3.0  # readout noise (e-)
    dcurrent = 3.0  # dark current (e-/pixel/s)
    nbiases = 11  # number of biases used in calibration
    tarea = 485000.0  # squared centimeters
    teldiam = 8.20  # diameter in meters
    h = 6.626196e-27  # erg.s
    if blueMUSE:
        spaxel = 0.3  # spaxel scale (arcsecs)
        fins = 0.2  # Instrument image quality (arcsecs)
        lmin = 3500.0  # minimum wavelength
        lmax = 6000.0  # maximum wavelength
        lstep = 0.66  # spectral sampling (Angstroms)
        lsf = lstep * 2.0  # in Angstroms
        musetrans = np.loadtxt(MUSE_etc_dir.joinpath("NewBlueMUSE_noatm.txt"))
        wmusetrans = musetrans[:, 0] * 10.0  # in Angstroms
        valmusetrans = musetrans[:, 1]
    else:
        spaxel = 0.2  # spaxel scale (arcsecs)
        fins = 0.15  # Instrument image quality (arcsecs)
        lmin = 4750.0  # minimum wavelength
        lmax = 9350.0  # maximum wavelength
        lstep = 1.25  # spectral sampling (Angstroms)
        lsf = 2.5  # in Angstroms
        musetrans = np.loadtxt(MUSE_etc_dir.joinpath("WFM_NONAO_N.dat.txt"))
        polysky = [
            -6.32655161e-12,
            1.94056813e-08,
            -2.25416420e-05,
            1.19349511e-02,
            -1.50077035e00,
        ]
        psky = np.polyval(
            polysky, musetrans[:, 0]
        )  # sky transmission fit over MUSE wavelength
        wmusetrans = musetrans[:, 0] * 10.0  # in Angstroms
        valmusetrans = musetrans[:, 1] / psky

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
    yy, xx = np.mgrid[: shape[0], : shape[1]]

    def moffat(p, q):
        xdiff = p - 50.0
        ydiff = q - 50.0
        return norm * (1 + (xdiff / a) ** 2 + (ydiff / a) ** 2) ** (-n)

    posmin = 50 - int(nspatial * spaxel / 2.0 / 0.05)  # in 0.05 arcsec pixels
    posmax = posmin + int(nspatial * spaxel / 0.05)  # in 0.05 arcsec pixels

    # For point sources compute the fraction of the total flux of the source
    if pointsource:
        for k in range(wrange.shape[0]):
            # All of this is based on ESO atmosphere turbulence model (ETC)
            ftel = 0.0000212 * wrange[k] / teldiam  # Diffraction limit in arcsec
            r0 = (
                0.100 * (seeing ** -1) * (wrange[k] / 5000.0) ** 1.2 * (airmass) ** -0.6
            )  # Fried parameter in meters
            fkolb = -0.981644
            if r0 < 5.4:
                fatm = (
                    seeing
                    * (airmass ** 0.6)
                    * (wrange[k] / 5000.0) ** (-0.2)
                    * np.sqrt((1.0 + fkolb * 2.183 * (r0 / 46.0) ** 0.356))
                )
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
    if moon == "g":
        skyemtable = np.loadtxt(
            MUSE_etc_dir.joinpath("radiance_airmass1.0_0.5moon.txt")
        )
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = (
            skyemtable[:, 1] * airmass
        )  # in photons / s / m2 / micron / arcsec2 approximated at given airmass
    else:  # dark conditions - no moon
        skyemtable = np.loadtxt(
            MUSE_etc_dir.joinpath("radiance_airmass1.0_newmoon.txt")
        )  # sky spectrum (grey) - 0.5 FLI
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = skyemtable[:, 1] * airmass  # in photons / s / m2 / micron / arcsec2
    # Interpolate sky spectrum at instrumental wavelengths
    sky = np.interp(wrange, skyemw, skyemflux)
    # loads sky transmission
    atmtrans = np.loadtxt(MUSE_etc_dir.joinpath("transmission_airmass1.txt"))
    atmtransw = atmtrans[:, 0] * 10.0  # In Angstroms
    atmtransval = atmtrans[:, 1]
    atm = np.interp(wrange, atmtransw, atmtransval)
    # Interpolate transmission including sky transmission at corresponding airmass
    # Note: ESO ETC includes a 60% margin for MUSE
    transm = np.interp(wrange, wmusetrans, valmusetrans) * (atm ** (airmass))
    transmnoatm = np.interp(wrange, wmusetrans, valmusetrans)

    dit = exptime
    ndit = nexp
    for k in range(wrange.shape[0]):
        kmin = 1 + np.max([-1, int(k - nspectral / 2)])
        kmax = 1 + np.min([wrange.shape[0] - 1, int(k + nspectral / 2)])

        if pointsource:
            signal = (
                (
                    np.sum(flux[kmin:kmax] * transm[kmin:kmax] * frac[kmin:kmax])
                    * lstep
                    / (h * 3e18 / wrange[k])
                )
                * tarea
                * dit
                * ndit
            )  # in electrons
        else:  # extended source, flux is per arcsec2
            signal = (
                (
                    np.sum(flux[kmin:kmax] * transm[kmin:kmax])
                    * lstep
                    / (h * 3e18 / wrange[k])
                )
                * tarea
                * dit
                * ndit
                * pixelarea
            )  # in electrons
        skysignal = (
            (np.sum(sky[kmin:kmax] * transmnoatm[kmin:kmax]) * lstep / 10000.0)
            * (tarea / 10000.0)
            * dit
            * ndit
            * pixelarea
        )  # lstep converted in microns, tarea in m2                                  #in electrons
        skyelectrons[k] = skysignal
        noise = np.sqrt(
            ron * ron * npixels * (1.0 + 1.0 / nbiases) * ndit
            + dcurrent * (dit * ndit / 3600.0) * npixels
            + signal
            + skysignal
        )  # in electrons
        snratio[k] = signal / noise
    return np.vstack([wrange, snratio])
