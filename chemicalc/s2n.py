import numpy as np
from scipy.interpolate import interp1d
import mechanicalsoup
from chemicalc import exception as e

keck_options = {'instrument': ['lris', 'deimos', 'hires'],
                'mag type': ['Vega', 'AB'],
                'filter': ['sdss_r.dat', 'sdss_g.dat', 'sdss_i.dat', 'sdss_u.dat', 'sdss_z.dat',
                           'Buser_B.dat', 'Buser_V.dat', 'Cousins_R.dat', 'Cousins_I.dat'],
                'template': ['O5V_pickles_1.fits', 'B5V_pickles_6.fits',
                             'A0V_pickles_9.fits', 'A5V_pickles_12.fits',
                             'F5V_pickles_16.fits', 'G5V_pickles_27.fits',
                             'K0V_pickles_32.fits', 'K5V_pickles_36.fits',
                             'M5V_pickles_44.fits'],
                'grating (DEIMOS)': ['600Z', '900Z', '1200G', '1200B'],
                'grating (LRIS)': ['600/7500', '600/10000', '1200/9000', '400/8500', '831/8200'],
                'grism (LRIS)': ['B300', 'B600'],
                'binning (DEIMOS)': ['1x1'],
                'binning (LRIS)': ['1x1', '2x1', '2x2', '3x1'],
                'binning (HIRES)': ['1x1', '2x1', '2x2', '3x1'],
                'slitwidth (DEIMOS)': ['0.75', '1.0', '1.5'],
                'slitwidth (LRIS)': ['0.7', '1.0', '1.5'],
                'slitwidth (HIRES)': ["C5", "E4", "B2", "B5", "E5", "D3"],
                'slitwidth arcsec (HIRES)': [1.15, 0.40, 0.57, 0.86, 0.80, 1.72],
                'dichroic (LRIS)': ['D560'],
                'central wavelength (DEIMOS)': ['5000', '6000', '7000', '8000'],
                }


class Sig2NoiseWMKO:
    def __init__(self, instrument, exptime, mag, template,
                 magtype='Vega', band='Cousins_I.dat',
                 airmass=1.1, seeing=0.75, redshift=0):
        if instrument not in keck_options['instrument']:
            raise e.S2NInputError(f"{instrument} not one of {keck_options['instrument']}")
        if magtype not in keck_options['mag type']:
            raise e.S2NInputError(f"{magtype} not one of {keck_options['mag type']}")
        if band not in keck_options['filter']:
            raise e.S2NInputError(f"{band} not one of {keck_options['filter']}")
        if template not in keck_options['template']:
            raise e.S2NInputError(f"{template} not one of {keck_options['template']}")
        self.instrument = instrument
        self.mag = mag
        self.magtype = magtype
        self.filter = band
        self.template = template
        self.exptime = exptime
        self.airmass = airmass
        self.seeing = seeing
        self.redshift = redshift


class Sig2NoiseDEIMOS(Sig2NoiseWMKO):
    def __init__(self, grating, exptime, mag, template,
                 magtype='Vega', band='Cousins_I.dat',
                 cwave='7000', slitwidth='0.75', binning='1x1',
                 airmass=1.1, seeing=0.75, redshift=0):
        Sig2NoiseWMKO.__init__(self, 'deimos', exptime, mag, template, magtype, band, airmass, seeing, redshift)
        if grating not in keck_options['grating (DEIMOS)']:
            raise e.S2NInputError(f"{grating} not one of {keck_options['grating (DEIMOS)']}")
        if binning not in keck_options['binning (DEIMOS)']:
            raise e.S2NInputError(f"{binning} not one of {keck_options['binning (DEIMOS)']}")
        if slitwidth not in keck_options['slitwidth (DEIMOS)']:
            raise e.S2NInputError(f"{slitwidth} not one of {keck_options['slitwidth (DEIMOS)']}")
        if cwave not in keck_options['central wavelength (DEIMOS)']:
            raise e.S2NInputError(f"{cwave} not one of {keck_options['central wavelength (DEIMOS)']}")
        self.grating = grating
        self.binning = binning
        self.slitwidth = slitwidth
        self.cwave = cwave

    def query_s2n(self, wavelength='default'):
        url = 'http://etc.ucolick.org/web_s2n/deimos'
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form['grating'] = self.grating
        form['cwave'] = self.cwave
        form['slitwidth'] = self.slitwidth
        form['binning'] = self.binning
        form['exptime'] = str(self.exptime)
        form['mag'] = str(self.mag)
        form['ffilter'] = self.filter
        if self.magtype.lower() == 'vega':
            form['mtype'] = '1'
        elif self.magtype.lower() == 'ab':
            form['mtype'] = '2'
        form['seeing'] = str(self.seeing)
        form['template'] = self.template
        form['airmass'] = str(self.airmass)
        form['redshift'] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data['s2n']).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == 'default':
            return snr
        else:
            raise e.S2NInputError("Wavelength input not recognized")


class Sig2NoiseLRIS(Sig2NoiseWMKO):
    def __init__(self, grating, grism, exptime, mag, template,
                 magtype='Vega', band='Cousins_I.dat',
                 dichroic='D560', slitwidth='0.7', binning='1x1',
                 airmass=1.1, seeing=0.75, redshift=0):
        Sig2NoiseWMKO.__init__(self, 'lris', exptime, mag, template, magtype, band, airmass, seeing, redshift)
        if grating not in keck_options['grating (LRIS)']:
            raise e.S2NInputError(f"{grating} not one of {keck_options['grating (LRIS)']}")
        if grism not in keck_options['grism (LRIS)']:
            raise e.S2NInputError(f"{grism} not one of {keck_options['grism (LRIS)']}")
        if binning not in keck_options['binning (LRIS)']:
            raise e.S2NInputError(f"{binning} not one of {keck_options['binning (LRIS)']}")
        if slitwidth not in keck_options['slitwidth (LRIS)']:
            raise e.S2NInputError(f"{slitwidth} not one of {keck_options['slitwidth (LRIS)']}")
        if dichroic not in keck_options['dichroic (LRIS)']:
            raise e.S2NInputError(f"{dichroic} not one of {keck_options['dichroic (LRIS)']}")
        self.grating = grating
        self.grism = grism
        self.binning = binning
        self.slitwidth = slitwidth
        self.dichroic = dichroic

    def query_s2n(self, wavelength='default'):
        url = 'http://etc.ucolick.org/web_s2n/lris'
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form['grating'] = self.grating
        form['grism'] = self.grism
        form['dichroic'] = self.dichroic
        form['slitwidth'] = self.slitwidth
        form['binning'] = self.binning
        form['exptime'] = str(self.exptime)
        form['mag'] = str(self.mag)
        form['ffilter'] = self.filter
        if self.magtype.lower() == 'vega':
            form['mtype'] = '1'
        elif self.magtype.lower() == 'ab':
            form['mtype'] = '2'
        form['seeing'] = str(self.seeing)
        form['template'] = self.template
        form['airmass'] = str(self.airmass)
        form['redshift'] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data['s2n']).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == 'default':
            return snr
        else:
            raise e.S2NInputError("Wavelength input not recognized")


class Sig2NoiseHIRES(Sig2NoiseWMKO):
    def __init__(self, slitwidth, exptime, mag, template,
                 magtype='Vega', band='Cousins_I.dat', binning='1x1',
                 airmass=1.1, seeing=0.75, redshift=0):
        Sig2NoiseWMKO.__init__(self, 'hires', exptime, mag, template, magtype, band, airmass, seeing, redshift)
        if binning not in keck_options['binning (HIRES)']:
            raise e.S2NInputError(f"{binning} not one of {keck_options['binning (HIRES)']}")
        if slitwidth not in keck_options['slitwidth (HIRES)']:
            raise e.S2NInputError(f"{slitwidth} not one of {keck_options['slitwidth (HIRES)']}")
        self.binning = binning
        self.slitwidth = slitwidth

    def query_s2n(self, wavelength='default'):
        url = 'http://etc.ucolick.org/web_s2n/hires'
        browser = mechanicalsoup.StatefulBrowser()
        browser.open(url)
        form = browser.select_form()
        form['slitwidth'] = self.slitwidth
        form['binning'] = self.binning
        form['exptime'] = str(self.exptime)
        form['mag'] = str(self.mag)
        form['ffilter'] = self.filter
        if self.magtype.lower() == 'vega':
            form['mtype'] = '1'
        elif self.magtype.lower() == 'ab':
            form['mtype'] = '2'
        form['seeing'] = str(self.seeing)
        form['template'] = self.template
        form['airmass'] = str(self.airmass)
        form['redshift'] = str(self.redshift)
        data = browser.submit_selected().json()
        snr = np.array(data['s2n']).T
        if type(wavelength) == np.ndarray:
            snr_interpolator = interp1d(snr[0], snr[1])
            return snr_interpolator(wavelength)
        elif wavelength == 'default':
            return snr
        else:
            raise e.S2NInputError("Wavelength input not recognized")
