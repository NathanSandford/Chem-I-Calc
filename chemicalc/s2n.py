import numpy as np
from scipy.interpolate import interp1d
import mechanicalsoup
import requests
import json
from chemicalc.utils import decode_base64_dict
from chemicalc.file_mgmt import data_dir

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
    "template": ["o5v", "o9v", "b1v", "b2ic", "b3v", "b8v", "b9iii", "b9v", "a0iii", "a0v", "a2v",
                 "f0v", "g0i", "g2v", "g5iii", "k2v", "k7v", "m2v" , "flat", "WD",
                 "LBG_EW_le_0", "LBG_EW_0_20", "LBG_EW_ge_20", "qso1", "qso2", "elliptical", "spiral_Sc", "HII", "PN"]

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
            raise KeyError(
                f"{instrument} not one of {keck_options['instrument']}"
            )
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
        raise NotImplementedError("No generic S/N query, see specific instrument children classes")


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
            raise KeyError(
                f"{grating} not one of {keck_options['grating (DEIMOS)']}"
            )
        if binning not in keck_options["binning (DEIMOS)"]:
            raise KeyError(
                f"{binning} not one of {keck_options['binning (DEIMOS)']}"
            )
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
            raise KeyError(
                f"{grating} not one of {keck_options['grating (LRIS)']}"
            )
        if grism not in keck_options["grism (LRIS)"]:
            raise KeyError(f"{grism} not one of {keck_options['grism (LRIS)']}")
        if binning not in keck_options["binning (LRIS)"]:
            raise KeyError(
                f"{binning} not one of {keck_options['binning (LRIS)']}"
            )
        if slitwidth not in keck_options["slitwidth (LRIS)"]:
            raise KeyError(
                f"{slitwidth} not one of {keck_options['slitwidth (LRIS)']}"
            )
        if dichroic not in keck_options["dichroic (LRIS)"]:
            raise KeyError(
                f"{dichroic} not one of {keck_options['dichroic (LRIS)']}"
            )
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
            raise KeyError(
                f"{binning} not one of {keck_options['binning (ESI)']}"
            )
        if slitwidth not in keck_options["slitwidth (ESI)"]:
            raise KeyError(
                f"{slitwidth} not one of {keck_options['slitwidth (ESI)']}"
            )
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
            raise KeyError(
                f"{binning} not one of {keck_options['binning (HIRES)']}"
            )
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

class Sig2NoiseMSE():
    def __init__(
            self,
            exptime,
            mag,
            template,
            spec_mode='LR',
            filter="g",
            airmass="1.2",
            seeing=0.5,
            skymag=20.7,
            src_type="point",
            redshift=0,
    ):
        self.url_base = 'http://etc-dev.cfht.hawaii.edu/cgi-bin/mse/mse_wrapper.py'
        # Hard Coded Values
        self.sessionID = 1234
        self.coating = 'ZeCoat'
        self.fibdiam = 1
        self.spatbin = 2
        self.specbin = 1
        self.meth = 'getSNR'
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

        def query_s2n(self, wavelength="default", smoothed=False,):
            url = f"{self.url_base}?" \
                  + f"sessionID={self.sessionID}&" \
                  + f"coating={self.coating}&" \
                  + f"seeing={self.seeing:1.2f}&" \
                  + f"airmass={self.airmass:1.1f}&" \
                  + f"skymag={self.skymag:2.1f}&" \
                  + f"spectro={self.spec_mode}&" \
                  + f"fibdiam={self.fibdiam}&" \
                  + f"spatbin={self.spatbin}&" \
                  + f"specbin={self.specbin}&" \
                  + f"meth={self.meth}&" \
                  + f"etime={self.exptime}&" \
                  + f"snr={self.snr_value}&" \
                  + f"src_type={self.src_type}&" \
                  + f"tgtmag={self.mag:2.1f}&" \
                  + f"redshift={self.redshift}&" \
                  + f"band={self.filter}&" \
                  + f"template={self.template}"
            response = requests.post(url)
            # Parse HTML response
            r = response.text.split('docs_json = \'')[1].split('\';')[0]
            model = json.loads(r)
            key = list(model.keys())[0]
            model_dict = model[key]
            model_pass1 = [_ for _ in model_dict['roots']['references'] if 'data' in _['attributes'].keys()]
            model_pass2 = [_ for _ in model_pass1 if '__ndarray__' in _['attributes']['data']['x']]
            x = {}
            y = {}
            for i, tmp in enumerate(model_pass2):
                x_str = tmp['attributes']['data']['x']
                x[i] = decode_base64_dict(x_str)
                y_str = tmp['attributes']['data']['y']
                y[i] = decode_base64_dict(y_str)
            # Sort Arrays
            order = np.argsort([array[0] for i, array in x.items()])
            x = {i: x[j] for i, j in enumerate(order)}
            y = {i: y[j] for i, j in enumerate(order)}
            x = {i: x[2 * i] for i in range(int(len(x) / 2))}
            if smoothed:
                y = {i: (y[2 * i] if (np.mean(y[2 * i]) > np.mean(y[2 * i + 1])) else y[2 * i + 1]) for i in
                     range(int(len(y) / 2))}
            else:
                y = {i: (y[2 * i] if (np.mean(y[2 * i]) < np.mean(y[2 * i + 1])) else y[2 * i + 1]) for i in
                     range(int(len(y) / 2))}
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
