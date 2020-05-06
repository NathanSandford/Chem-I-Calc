from typing import Optional, Union, Tuple
from warnings import warn
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import mechanicalsoup
import requests
import json
from chemicalc.utils import decode_base64_dict
from chemicalc.file_mgmt import etc_file_dir, download_bluemuse_files


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
vlt_options = {
    "instruments": ['UVES', 'FLAMES-UVES', 'FLAMES-GIRAFFE', 'XSHOOTER', 'MUSE'],
    "src_target_mag_band": ['U',  # NOT MUSE
                            'B', 'V', 'R', 'I',
                            'J', 'H', 'K',  # X-SHOOTER ONLY
                            'sloan_g_prime', 'sloan_r_prime', 'sloan_i_prime', 'sloan_z_prime'],  # MUSE ONLY
    "src_target_mag_system": ['Vega', 'AB'],
    "src_target_type": ["template_spectrum"],
    "src_target_spec_type": ["Pickles_O5V",
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
                             "qso-interp"],
    "uves_det_cd_name": ["Blue_346", "Blue_437", "red__520", "red__580", "red__600", "red__860",
                         "Dicroic1_Blue_346", "Dicroic2_Blue_346", "Dicroic1_red__580",
                         "Dicroic1_Blue_390", "Dicroic2_Blue_390", "Dicroic1_red__564",
                         "Dicroic2_Blue_437", "Dicroic2_red__760", "Dicroic2_red__860"],
    "uves_slit_width": ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1",
                        "1.2", "1.5", "1.8", "2.1", "2.4", "2.7", "3.0", "5.0", "10.0"],
    "uves_ccd_binning": ['1x1', '1x1v', '2x2', '2x1', '3x2'],
    "giraffe_sky_sampling_mode": ['MEDUSA', 'IFU052', 'ARGUS052', 'ARGUS030',],
    "giraffe_resolution": ['HR', 'LR'],
    "giraffe_slicer_hr": ['HR01', 'HR02', 'HR03', 'HR04', 'HR05A', 'HR05B', 'HR06', 'HR07A', 'HR07B', 'HR08',
                          'HR09A', 'HR09B', 'HR10', 'HR11', 'HR12', 'HR13', 'HR014A', 'HR014B', 'HR15', 'HR15n',
                          'HR16', 'HR17A', 'HR17B', 'HR17B', 'HR18', 'HR19A', 'HR19B', 'HR20A', 'HR20B', 'HR21',
                          'HR22A', 'HR22B'],
    "giraffe_slicer_lr": ['LR01', 'LR02', 'LR03', 'LR04', 'LR05', 'LR06', 'LR07', 'LR08'],
    "giraffe_ccd_mode": ['standard', 'fast', 'slow'],
    "xshooter_uvb_slit_width": ["0.5", "0.8", "1.0", "1.3", "1.6", "5.0"],
    "xshooter_vis_slit_width": ["0.4", "0.7", "0.9", "1.2", "1.5", "5.0"],
    "xshooter_nir_slit_width": ["0.4", "0.6", "0.9", "1.2", "1.5", "5.0"],
    "xshooter_uvb_ccd_binning": ["high1x1slow", "high1x2slow", "high2x2slow",
                                 "low1x1fast", "low1x2fast", "low2x2fast"],
    "xshooter_vis_ccd_binning": ["high1x1slow", "high1x2slow", "high2x2slow",
                                 "low1x1fast", "low1x2fast", "low2x2fast"],
    "muse_setting": ['WFM_NONAO_N',  # Wide Field Mode without AO, nominal  wavelength range
                     'WFM_NONAO_E',  # Wide Field Mode without AO, extended wavelength range
                     'WFM_AO_N',  # Wide Field Mode with AO, nominal wavelength range
                     'WFM_AO_E',  # Wide Field Mode with AO, extended wavelength range
                     'NFM_AO_N'],  # Narrow Field Mode with AO, nominal wavelength range
    "muse_spatial_binning": ['1', '2', '3', '4', '5', '10', '30', '60', '100'],
    "muse_spectra_binning": ['1', '2', '3', '4', '5', '10', '20', '30', '40',
                              '50', '100', '200', '400', '800', '1600', '3200'],
}


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


class Sig2NoiseVLT:
    # TODO: Refactor to be consistent with WMKO ETC query.
    # TODO: Implement MARCS stellar template selection
    def __init__(self, instrument: str,
                 exptime: float,
                 src_target_mag: float,
                 src_target_mag_band: str = 'V',
                 src_target_mag_system: str = 'Vega',
                 src_target_type: str = 'template_spectrum',
                 src_target_spec_type: str = 'Pickles_K2V',
                 src_target_redshift: float = 0,
                 sky_airmass: float = 1.1,
                 sky_moon_fli: float = 0.0,
                 sky_seeing_iq: float = 0.75,
                 uves_det_cd_name="Red__580",
                 uves_slit_width="1.0",
                 uves_ccd_binning='1x1',
                 giraffe_sky_sampling_mode='MEDUSA',
                 giraffe_fiber_obj_decenter=0.0,
                 giraffe_resolution='HR',
                 giraffe_slicer_hr='HR10',
                 giraffe_slicer_lr='LR08',
                 giraffe_ccd_mode='standard',
                 xshooter_uvb_slit_width='0.8',
                 xshooter_vis_slit_width='0.7',
                 xshooter_nir_slit_width='0.9',
                 xshooter_uvb_ccd_binning="high1x1slow",
                 xshooter_vis_ccd_binning="high1x1slow",
                 muse_setting='WFM_NONAO_N',
                 muse_spatial_binning='3',
                 muse_spectra_binning='1',
                 muse_target_offset=0,
                 **kwargs):
        if instrument not in vlt_options["instruments"]:
            raise KeyError(f"{instrument} not one of {vlt_options['instruments']}")
        self.instrument = instrument
        self.urls = {"UVES": "http://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES++INS.MODE=spectro",
                     "FLAMES-UVES": "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=UVES+INS.MODE=FLAMES",
                     "FLAMES-GIRAFFE": "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=GIRAFFE+INS.MODE=spectro",
                     "MUSE": "https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=MUSE+INS.MODE=swspectr"}
        self.url =  self.urls[instrument]
        if not exptime > 0:
            raise ValueError("Exposure Time must be positive")
        self.exptime = exptime
        self.src_target_mag = src_target_mag
        if src_target_mag_band not in vlt_options["src_target_mag_band"]:
            raise KeyError(f"{src_target_mag_band} not one of {vlt_options['src_target_mag_band']}")
        if src_target_mag_band in ['J', 'H', 'K'] and instrument != 'X-SHOOTER':
            raise KeyError("J, H, and K bands are only valid for X-SHOOTER")
        if src_target_mag_band in ['sloan_g_prime', 'sloan_r_prime', 'sloan_i_prime', 'sloan_z_prime'] and instrument != 'MUSE':
            raise KeyError("g', r', i', and z' bands are only valid for MUSE")
        if src_target_mag_band == 'U' and instrument == 'MUSE':
            raise KeyError("U bands is not valid for MUSE")
        self.src_target_mag_band = src_target_mag_band
        if src_target_mag_system not in vlt_options["src_target_mag_system"]:
            raise KeyError(f"{src_target_mag_system} not one of {vlt_options['src_target_mag_system']}")
        self.src_target_mag_system = src_target_mag_system
        if src_target_type not in vlt_options["src_target_type"]:
            raise KeyError(f"Only {vlt_options['src_target_type']} is supported currently")
        self.src_target_type = src_target_type
        if src_target_spec_type not in vlt_options["src_target_spec_type"]:
            raise KeyError(f"{src_target_spec_type} not one of {vlt_options['src_target_spec_type']}")
        self.src_target_spec_type = src_target_spec_type
        if not src_target_redshift >= 0:
            raise ValueError("Redshift must be positive")
        self.src_target_redshift = src_target_redshift
        if not sky_airmass >= 1.0:
            raise ValueError("Airmass must be > 1.0")
        self.sky_airmass = sky_airmass
        if sky_moon_fli >= 0.0 or sky_moon_fli <= 1.0:
            raise ValueError('sky_moon_fli must be between 0.0 (new) and 1.0 (full)')
        self.sky_moon_fli = sky_moon_fli
        if sky_seeing_iq > 3:
            raise ValueError("Seeing must be < 3 arcsec")
        self.sky_seeing_iq = sky_seeing_iq
        if uves_det_cd_name not in vlt_options["uves_det_cd_name"]:
            raise KeyError(f"{uves_det_cd_name} not one of {vlt_options['uves_det_cd_name']}")
        self.uves_det_cd_name = uves_det_cd_name
        if uves_slit_width not in vlt_options["uves_slit_width"]:
            raise KeyError(f"{uves_slit_width} not one of {vlt_options['uves_slit_width']}")
        self.uves_slit_width = uves_slit_width
        if uves_ccd_binning not in vlt_options["uves_ccd_binning"]:
            raise KeyError(f"{uves_ccd_binning} not one of {vlt_options['uves_ccd_binning']}")
        self.uves_ccd_binning = uves_ccd_binning
        if giraffe_sky_sampling_mode not in vlt_options["giraffe_sky_sampling_mode"]:
            raise KeyError(f"{giraffe_sky_sampling_mode} not one of {vlt_options['giraffe_sky_sampling_mode']}")
        self.giraffe_sky_sampling_mode = giraffe_sky_sampling_mode
        if giraffe_fiber_obj_decenter <= 0:
            raise ValueError("giraffe_fiber_obj_decenter must be positive")
        self.giraffe_fiber_obj_decenter = giraffe_fiber_obj_decenter
        if giraffe_resolution not in vlt_options["giraffe_resolution"]:
            raise KeyError(f"{giraffe_resolution} not one of {vlt_options['giraffe_resolution']}")
        self.giraffe_resolution = giraffe_resolution
        if giraffe_slicer_hr not in vlt_options["giraffe_slicer_hr"]:
            raise KeyError(f"{giraffe_slicer_hr} not one of {vlt_options['giraffe_slicer_hr']}")
        self.giraffe_slicer_hr = giraffe_slicer_hr
        if giraffe_slicer_lr not in vlt_options["giraffe_slicer_lr"]:
            raise KeyError(f"{giraffe_slicer_lr} not one of {vlt_options['giraffe_slicer_lr']}")
        self.giraffe_slicer_lr = giraffe_slicer_lr
        if giraffe_ccd_mode not in vlt_options["giraffe_ccd_mode"]:
            raise KeyError(f"{giraffe_ccd_mode} not one of {vlt_options['giraffe_ccd_mode']}")
        self.giraffe_ccd_mode = giraffe_ccd_mode
        if xshooter_uvb_slit_width not in vlt_options["xshooter_uvb_slit_width"]:
            raise KeyError(f"{xshooter_uvb_slit_width} not one of {vlt_options['xshooter_uvb_slit_width']}")
        self.xshooter_uvb_slit_width = xshooter_uvb_slit_width
        if xshooter_vis_slit_width not in vlt_options["xshooter_vis_slit_width"]:
            raise KeyError(f"{xshooter_vis_slit_width} not one of {vlt_options['xshooter_vis_slit_width']}")
        self.xshooter_vis_slit_width = xshooter_vis_slit_width
        if xshooter_nir_slit_width not in vlt_options["xshooter_nir_slit_width"]:
            raise KeyError(f"{xshooter_nir_slit_width} not one of {vlt_options['xshooter_nir_slit_width']}")
        self.xshooter_nir_slit_width = xshooter_nir_slit_width
        if xshooter_uvb_ccd_binning not in vlt_options["xshooter_uvb_ccd_binning"]:
            raise KeyError(f"{xshooter_uvb_ccd_binning} not one of {vlt_options['xshooter_uvb_ccd_binning']}")
        self.xshooter_uvb_ccd_binning = xshooter_uvb_ccd_binning
        if xshooter_vis_ccd_binning not in vlt_options["xshooter_vis_ccd_binning"]:
            raise KeyError(f"{xshooter_vis_ccd_binning} not one of {vlt_options['xshooter_vis_ccd_binning']}")
        self.xshooter_vis_ccd_binning = xshooter_vis_ccd_binning
        if muse_setting not in vlt_options["muse_setting"]:
            raise KeyError(f"{muse_setting} not one of {vlt_options['muse_setting']}")
        self.muse_setting = muse_setting
        if muse_spatial_binning not in vlt_options["muse_spatial_binning"]:
            raise KeyError(f"{muse_spatial_binning} not one of {vlt_options['muse_spatial_binning']}")
        self.muse_spatial_binning = muse_spatial_binning
        if muse_spectra_binning not in vlt_options["muse_spectra_binning"]:
            raise KeyError(f"{muse_spectra_binning} not one of {vlt_options['muse_spectra_binning']}")
        self.muse_spectra_binning = muse_spectra_binning
        if muse_target_offset <= 0:
            raise ValueError("muse_target_offset must be positive")
        self.muse_target_offset = muse_target_offset

    def query_s2n(self, uves_mid_order_only=False):
        url = self.urls[self.instrument]
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form["POSTFILE.FLAG"] = 0
        # Source Parameters
        form["SRC.TARGET.MAG"] = self.src_target_mag
        form.new_control(type='select', name="SRC.TARGET.MAG.BAND", value='V')
        form["SRC.TARGET.MAG.BAND"] = self.src_target_mag_band
        form["SRC.TARGET.MAG.SYSTEM"]  = self.src_target_mag_system
        form["SRC.TARGET.TYPE"] = self.src_target_type
        form["SRC.TARGET.SPEC.TYPE"] =  self.src_target_spec_type
        form["SRC.TARGET.REDSHIFT"] = self.src_target_redshift
        form["SRC.TARGET.GEOM"] = 'seeing_ltd'
        # Sky Parameters
        form["SKY.AIRMASS"] = self.sky_airmass
        form["SKY.MOON.FLI"] = self.sky_moon_fli
        form["USR.SEEING.OR.IQ"] = "iq_given"
        form["SKY.SEEING.IQ"] = self.sky_seeing_iq
        # Default Sky Background
        form["almanac_time_option"] = "almanac_time_option_ut_time"
        form["SKYMODEL.TARGET.ALT"] = 65.38
        form["SKYMODEL.MOON.SUN.SEP"] = 0
        if self.instrument == "UVES":
            self.uves_mid_order_only = uves_mid_order_only
            form["INS.NAME"] = 'UVES'
            form["INS.MODE"] = 'spectro'
            form["INS.PRE_SLIT.FILTER.NAME"] = 'ADC'
            form["INS.IMAGE_SLICERS.NAME"] = "None"
            form['INS.BELOW_SLIT.FILTER.NAME'] = 'NONE'
            form["INS.DET.SPECTRAL_FORMAT.NAME"] = "STANDARD"
            form["INS.DET.CD.NAME"] = self.uves_det_cd_name
            form["INS.SLIT.FROM_USER.WIDTH.VAL"] = self.uves_slit_width
            form["INS.DET.CCD.BINNING.VAL"] = self.uves_ccd_binning
            form["INS.DET.EXP.TIME.VAL"] = self.exptime
            form["INS.GEN.TABLE.SF.SWITCH.VAL"] = "yes"
            form["INS.GEN.TABLE.RES.SWITCH.VAL"] = "yes"
            form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        if self.instrument == "FLAMES-UVES":
            self.uves_mid_order_only = uves_mid_order_only
            form["INS.NAME"] = 'UVES'
            form["INS.MODE"] = 'FLAMES'
            form["INS.DET.CD.NAME"] = self.uves_det_cd_name
            form["INS.DET.EXP.TIME.VAL"] = self.exptime
            form["INS.GEN.TABLE.SF.SWITCH.VAL"] = "yes"
            form["INS.GEN.TABLE.RES.SWITCH.VAL"] = "yes"
            form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        if self.instrument == "FLAMES-GIRAFFE":
            form["INS.NAME"] = 'GIRAFFE'
            form["INS.MODE"] = 'spectro'
            form["INS.SKY.SAMPLING.MODE"] = self.giraffe_sky_sampling_mode
            form["INS.GIRAFFE.FIBER.OBJ.DECENTER"] = self.giraffe_fiber_obj_decenter
            form["INS.GIRAFFE.RESOLUTION"] = self.giraffe_resolution
            form["INS.IMAGE.SLICERS.NAME.HR"] = self.giraffe_slicer_hr
            form["INS.IMAGE.SLICERS.NAME.LR"] = self.giraffe_slicer_lr
            form["DET.CCD.MODE"] = self.giraffe_ccd_mode
            form["USR.OUT.MODE"] = "USR.OUT.MODE.EXPOSURE.TIME"
            form["USR.OUT.MODE.EXPOSURE.TIME"] = self.exptime
            form["USR.OUT.DISPLAY.SN.V.WAVELENGTH"] = "1"
        if self.instrument == 'X-SHOOTER':
            form["INS.NAME"] = 'X-SHOOTER'
            form["INS.MODE"] = 'spectro'
            form["INS.ARM.UVB.FLAG"] = "1"
            form["INS.ARM.VIS.FLAG"] = "1"
            form["INS.ARM.NIR.FLAG"] = "1"
            form["INS.SLIT.FROM_USER.WIDTH.VAL.UVB"] = self.xshooter_uvb_slit_width
            form["INS.SLIT.FROM_USER.WIDTH.VAL.VIS"] = self.xshooter_vis_slit_width
            form["INS.SLIT.FROM_USER.WIDTH.VAL.NIR"] = self.xshooter_nir_slit_width
            form["INS.DET.DIT.UVB"] = self.exptime
            form["INS.DET.DIT.VIS"] = self.exptime
            form["INS.DET.DIT.NIR"] = self.exptime
            form["INS.DET.CCD.BINNING.VAL.UVB"] = self.xshooter_uvb_ccd_binning
            form["INS.DET.CCD.BINNING.VAL.VIS"] = self.xshooter_vis_ccd_binning
            form["INS.GEN.GRAPH.S2N.SWITCH.VAL"] = "yes"
        if self.instrument == 'MUSE':
            form["INS.NAME"] = 'MUSE'
            form["INS.MODE"] = 'swspectr'
            form["INS.MUSE.SETTING.KEY"] = self.muse_setting
            form["INS.MUSE.SPATIAL.NPIX.LINEAR"] = self.muse_spatial_binning
            form["INS.MUSE.SPECTRAL.NPIX.LINEAR"] = self.muse_spectra_binning
            form["SRC.TARGET.GEOM.DISTANCE"] = self.muse_target_offset
            form["USR.OBS.SETUP.TYPE"] = 'givenexptime'
            form["DET.IR.NDIT"] = 1
            form["DET.IR.DIT"] = self.exptime
            form["USR.OUT.DISPLAY.SN.V.WAVELENGTH"] = 1
        self.data = browser.submit_selected()
        if self.instrument in ["UVES", "FLAMES-UVES"]:
            return self.parse_uves_etc()
        elif self.instrument == 'X-SHOOTER':
            return self.parse_xshooter_etc()
        else:
            return self.parse_basic_etc()


    def parse_uves_etc(self):
        if self.uves_mid_order_only:
            snr_url1 = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
            snr_url2 = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[2].split('" TITLE')[0]
            snr_txt1 = requests.post(snr_url1).text
            snr_txt2 = requests.post(snr_url2).text
            snr1 = pd.DataFrame([row.split('\t') for row in snr_txt1.split('\n')[:-1]])
            snr2 = pd.DataFrame([row.split('\t') for row in snr_txt2.split('\n')[:-1]])
            uves_snr = pd.concat([snr1, snr2])
            uves_snr.index = uves_snr.pop(0)
            uves_snr.sort_index(inplace=True)
            uves_snr = np.vstack([uves_snr.index.values, uves_snr[1].values]).astype(float)
        else:
            mit_tab1 = pd.read_html(
                '<table class="echelleTable' + self.data.text.split('<table class="echelleTable')[1].split('</table>')[0])[0]
            mit_tab1.columns = mit_tab1.loc[0]
            mit_tab1.drop(0, axis=0, inplace=True)
            mit_tab2 = pd.read_html(
                '<table class="echelleTable' + self.data.text.split('<table class="echelleTable')[2].split('</table>')[0])[0]
            mit_tab2.columns = mit_tab2.loc[1]
            mit_tab2.drop([0, 1], axis=0, inplace=True)
            eev_tab1 = pd.read_html(
                '<table class="echelleTable' + self.data.text.split('<table class="echelleTable')[3].split('</table>')[0])[0]
            eev_tab1.columns = eev_tab1.loc[0]
            eev_tab1.drop(0, axis=0, inplace=True)
            eev_tab2 = pd.read_html(
                '<table class="echelleTable' + self.data.text.split('<table class="echelleTable')[4].split('</table>')[0])[0]
            eev_tab2.columns = eev_tab2.loc[1]
            eev_tab2.drop([0, 1], axis=0, inplace=True)
            mit_wave_mid = mit_tab1['wav of central column (nm)']
            mit_wave_min = mit_tab1['FSR l Min (nm)']
            mit_wave_max = mit_tab1['FSR l Max (nm)']
            mit_snr_min = mit_tab2['S/N*'].iloc[:, 0]
            mit_snr_mid = mit_tab2['S/N*'].iloc[:, 1]
            mit_snr_max = mit_tab2['S/N*'].iloc[:, 2]
            eev_wave_mid = eev_tab1['wav of central column (nm)']
            eev_wave_min = eev_tab1['FSR l Min (nm)']
            eev_wave_max = eev_tab1['FSR l Max (nm)']
            eev_snr_min = eev_tab2['S/N*'].iloc[:, 0]
            eev_snr_mid = eev_tab2['S/N*'].iloc[:, 1]
            eev_snr_max = eev_tab2['S/N*'].iloc[:, 2]
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
            uves_snr = np.vstack([uves_snr.index.values, uves_snr.iloc[:].values]).astype(float)
        uves_snr[0] *= 10
        return uves_snr

    def parse_basic_etc(self):
        snr_url = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        snr_txt = requests.post(snr_url).text
        snr = pd.DataFrame([row.split(' ') for row in snr_txt.split('\n')[:-1]])
        snr.index = snr.pop(0)
        snr.sort_index(inplace=True)
        snr = np.vstack([snr.index.values, snr[1].values]).astype(float)
        snr[0] *= 10
        return snr

    def parse_xshooter_etc(self):
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

        snr_url1_mid = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[1].split('" TITLE')[0]
        snr_url2_mid = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[2].split('" TITLE')[0]
        snr_url3_mid = 'https://www.eso.org' + self.data.text.split('ASCII DATA INFO: URL="')[3].split('" TITLE')[0]
        snr_url1_max = snr_url1_mid[:-4] + '_FSRmax.dat'
        snr_url2_max = snr_url2_mid[:-4] + '_FSRmax.dat'
        snr_url3_max = snr_url3_mid[:-4] + '_FSRmax.dat'
        snr_url1_min = snr_url1_mid[:-4] + '_FSRmin.dat'
        snr_url2_min = snr_url2_mid[:-4] + '_FSRmin.dat'
        snr_url3_min = snr_url3_mid[:-4] + '_FSRmin.dat'
        snr_txt1_mid = requests.post(snr_url1_mid).text
        snr_txt2_mid = requests.post(snr_url2_mid).text
        snr_txt3_mid = requests.post(snr_url3_mid).text
        snr_txt1_max = requests.post(snr_url1_max).text
        snr_txt2_max = requests.post(snr_url2_max).text
        snr_txt3_max = requests.post(snr_url3_max).text
        snr_txt1_min = requests.post(snr_url1_min).text
        snr_txt2_min = requests.post(snr_url2_min).text
        snr_txt3_min = requests.post(snr_url3_min).text
        snr1_mid_df = pd.DataFrame([row.split('\t') for row in snr_txt1_mid.split('\n')[:-1]], dtype=float)
        snr2_mid_df = pd.DataFrame([row.split('\t') for row in snr_txt2_mid.split('\n')[:-1]], dtype=float)
        snr3_mid_df = pd.DataFrame([row.split('\t') for row in snr_txt3_mid.split('\n')[:-1]], dtype=float)
        snr1_max_df = pd.DataFrame([row.split('\t') for row in snr_txt1_max.split('\n')[:-1]], dtype=float)
        snr2_max_df = pd.DataFrame([row.split('\t') for row in snr_txt2_max.split('\n')[:-1]], dtype=float)
        snr3_max_df = pd.DataFrame([row.split('\t') for row in snr_txt3_max.split('\n')[:-1]], dtype=float)
        snr1_min_df = pd.DataFrame([row.split('\t') for row in snr_txt1_min.split('\n')[:-1]], dtype=float)
        snr2_min_df = pd.DataFrame([row.split('\t') for row in snr_txt2_min.split('\n')[:-1]], dtype=float)
        snr3_min_df = pd.DataFrame([row.split('\t') for row in snr_txt3_min.split('\n')[:-1]], dtype=float)
        snr1 = combine_xshooter_snr(snr1_min_df, snr1_mid_df, snr1_max_df, offset=1)
        snr2 = combine_xshooter_snr(snr2_min_df, snr2_mid_df, snr2_max_df, offset=0)
        snr3 = combine_xshooter_snr(snr3_min_df, snr3_mid_df, snr3_max_df, offset=1)
        snr = pd.concat([snr1, snr2, snr3])
        snr = np.vstack([snr.index.values, snr[1].values])
        snr[0] *= 10
        return snr


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
        S_red = 10 ** log_S_red * slit_loss_factor * A_per_pix_red
        S_blue = 10 ** log_S_blue * slit_loss_factor * A_per_pix_blue
        snr_red = (
            g_red * S_red / np.sqrt(g_red * S_red + sigma_RO_red ** 2)
        )
        snr_blue = (
            g_blue
            * S_blue
            / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
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
            S_red = 10 ** log_S_red * slit_loss_factor * A_per_pix_red
            snr = (
                g_red
                * S_red
                / np.sqrt(g_red * S_red + sigma_RO_red ** 2)
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
            S_blue = 10 ** log_S_blue * slit_loss_factor * A_per_pix_blue
            snr = (
                g_blue
                * S_blue
                / np.sqrt(g_blue * S_blue + sigma_RO_blue ** 2)
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
    spot_fwhm: float = 5.8,
    spec_res_indx: Optional[Union[str, float]] = None,
    spec_res_value: Optional[float] = None,
    spec_table: Optional[Union[str, float]] = None,
    mag: float = 24.0,
    mag_band: str = "g",
    mag_system: str = "AB",
    sky_mag: Optional[float] = None,
    sky_mag_band: str = 'g',
    sky_mag_system: str = 'AB',
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
    try:
        from enyo.etc import spectrum, efficiency, telescopes, aperture, detector, extract
        from enyo.etc.observe import Observation
        from enyo.scripts.fobos_etc import (
            get_wavelength_vector,
            read_emission_line_database,
            get_spectrum,
            get_sky_spectrum,
            get_source_distribution,
        )
    except ImportError:
        raise ImportError('To calculate FOBOS S/N you must first install the FOBOS ETC.\n '+
                          'See <> for installation instructions.')
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
    sky_spectrum = get_sky_spectrum(sky_mag, mag_band=sky_mag_band, mag_system=sky_mag_system)
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


def calculate_muse_snr(wave, flux, exptime, nexp, blueMUSE=False,
                       airmass=1.0, seeing=0.8, moon='d', pointsource=True,
                       nspatial=3, nspectral=1):
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
    MUSE_etc_dir = etc_file_dir.joinpath('MUSE')
    muse_files = [MUSE_etc_dir.joinpath('NewBlueMUSE_noatm.txt'),
                  MUSE_etc_dir.joinpath('radiance_airmass1.0_0.5moon.txt'),
                  MUSE_etc_dir.joinpath('radiance_airmass1.0_newmoon.txt'),
                  MUSE_etc_dir.joinpath('transmission_airmass1.txt'),
                  MUSE_etc_dir.joinpath('WFM_NONAO_N.dat.txt')]
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
        musetrans = np.loadtxt(MUSE_etc_dir.joinpath('NewBlueMUSE_noatm.txt'))
        wmusetrans = musetrans[:, 0] * 10.0  # in Angstroms
        valmusetrans = musetrans[:, 1]
    else:
        spaxel = 0.2  # spaxel scale (arcsecs)
        fins = 0.15  # Instrument image quality (arcsecs)
        lmin = 4750.0  # minimum wavelength
        lmax = 9350.0  # maximum wavelength
        lstep = 1.25  # spectral sampling (Angstroms)
        lsf = 2.5  # in Angstroms
        musetrans = np.loadtxt(MUSE_etc_dir.joinpath('WFM_NONAO_N.dat.txt'))
        polysky = [-6.32655161e-12, 1.94056813e-08, -2.25416420e-05, 1.19349511e-02, -1.50077035e+00]
        psky = np.polyval(polysky, musetrans[:, 0])  # sky transmission fit over MUSE wavelength
        wmusetrans = musetrans[:,0] * 10.0    #in Angstroms
        valmusetrans = musetrans[:,1] / psky

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
        skyemtable = np.loadtxt(MUSE_etc_dir.joinpath('radiance_airmass1.0_0.5moon.txt'))
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = skyemtable[:, 1] * airmass  # in photons / s / m2 / micron / arcsec2 approximated at given airmass
    else:  # dark conditions - no moon
        skyemtable = np.loadtxt(MUSE_etc_dir.joinpath('radiance_airmass1.0_newmoon.txt'))  # sky spectrum (grey) - 0.5 FLI
        skyemw = skyemtable[:, 0] * 10.0  # in Angstroms
        skyemflux = skyemtable[:, 1] * airmass  # in photons / s / m2 / micron / arcsec2
    # Interpolate sky spectrum at instrumental wavelengths
    sky = np.interp(wrange, skyemw, skyemflux)
    # loads sky transmission
    atmtrans = np.loadtxt(MUSE_etc_dir.joinpath('transmission_airmass1.txt'))
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
