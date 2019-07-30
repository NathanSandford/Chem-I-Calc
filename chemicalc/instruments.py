import numpy as np
from scipy.interpolate import interp1d
from chemicalc.utils import generate_wavelength_template
from chemicalc.exception import InstError


sampX = 3  # Placeholder for Unknown Wavelength Sampling


class InstConfig:
    def __init__(self, name: str, res: float, samp: float, start: float, end: float):
        self.name = name
        self.R_res = res
        self.R_samp = samp
        self.start_wavelength = start
        self.end_wavelength = end
        self.wave = generate_wavelength_template(self.start_wavelength, self.end_wavelength,
                                                 self.R_res * self.R_samp, truncate=False)
        self.snr = 100 * np.ones_like(self.wave)

    def set_wave(self):
        self.wave = generate_wavelength_template(self.start_wavelength, self.end_wavelength,
                                                 self.R_res * self.R_samp, truncate=False)

    def get_wave(self, truncate=False):
        wave = generate_wavelength_template(self.start_wavelength, self.end_wavelength,
                                            self.R_res*self.R_samp, truncate)
        self.wave = wave
        return wave

    def set_snr(self, snr_input):
        if (type(snr_input) == int) or (type(snr_input) == float):
            self.snr = snr_input * np.ones_like(self.wave)
        elif (type(snr_input) == np.ndarray) and (snr_input.ndim == 2):
            snr_interpolator = interp1d(snr_input[0], snr_input[1],
                                        bounds_error=False, fill_value='extrapolate')
            self.snr = snr_interpolator(self.wave)
        elif (type(snr_input) == np.ndarray) and (snr_input.ndim == 1):
            fake_wave = np.linspace(self.wave.min(), self.wave.max(), snr_input.shape[0])
            snr_interpolator = interp1d(fake_wave, snr_input,
                                        bounds_error=False, fill_value='extrapolate')
            self.snr = snr_interpolator(self.wave)
        else:
            self.snr = snr_input.query_s2n(wavelength=self.wave)

    def summary(self):
        print(f'{self.name}\n' +
              f'{self.start_wavelength} < lambda (A) < {self.end_wavelength}\n' +
              f'R ~ {self.R_res}\n' +
              f'Sampling ~ {self.R_samp} pix/FWHM')


class DEIMOS(InstConfig):
    def __init__(self, name, res, samp, start, end):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = 'WMKO'
        self.instrument = 'DEIMOS'
        self.mode = 'Multi-Object Spectrograph'
        self.snr = []


class LRIS(InstConfig):
    def __init__(self, name, res, samp, start, end):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = 'WMKO'
        self.instrument = 'LRIS'
        self.mode = 'Multi-Object Spectrograph'


class HIRES(InstConfig):
    def __init__(self, name, res, samp, start, end):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = 'WMKO'
        self.instrument = 'HIRES'
        self.mode = 'Echelle Spectrograph'


class MIKE(InstConfig):
    def __init__(self, name, res, samp, start, end):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = 'Magellan'
        self.instrument = 'MIKE'
        self.mode = 'Echelle Spectrograph'


class M2FS(InstConfig):
    def __init__(self, name, res, samp, start, end):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = 'Magellan'
        self.instrument = 'M2FS'
        self.mode = 'Multi-Object Spectrograph'


class MiscInstrument(InstConfig):
    def __init__(self, name, res, samp, start, end, facility=None, instrument=None, mode=None):
        InstConfig.__init__(self, name, res, samp, start, end)
        self.facility = facility
        self.instrument = instrument
        self.mode = mode


class FilterSet:
    def __init__(self, name, wave_eff=None, width=None, filters=None):
        self.name = name
        if wave_eff and width:
            if len(wave_eff) != len(width):
                raise InstError('Must provide same number of effective wavelengths and widths')
            self.wave_eff = wave_eff
            self.fwhm = width
            wave, throughput = generate_tophat_throughput(wave_eff, width)
            self.wave = wave
            self.throughput = throughput
        elif (wave_eff is True) != (width is True):
            raise InstError('Self-defined filters must include both wave_eff and fwhm')
        elif filters == 0:
            raise InstError('Must provide at least one filter')
        else:
            throughput, wave_eff, fwhm = load_throughput(filters=filters)
            self.throughput = throughput
            self.wave_eff = wave_eff
            self.fwhm = fwhm


class AllInstruments:
    def __init__(self):
        self.spectrographs = {  # # # WMKO # # #
                              'DEIMOS 1200G': DEIMOS('DEIMOS 1200G', res=6500, samp=4, start=6500, end=9000),
                              'DEIMOS 900ZD': DEIMOS('DEIMOS 900ZD', res=2550, samp=5, start=4000, end=7200),
                              'DEIMOS 1200B': DEIMOS('DEIMOS 1200B', res=4000, samp=4, start=4000, end=6400),
                              'LRIS 600/4000 (b)': LRIS('LRIS 600/4000 (b)', res=1800, samp=4, start=3900, end=5500),
                              'LRIS 1200/7500 (r)': LRIS('LRIS 1200/7500 (r)', res=4000, samp=5, start=7700, end=9000),
                              'HIRESr 1.0"': HIRES('HIRESr 1.0"', res=34000, samp=sampX, start=3900, end=8350),
                              'HIRESr 0.8"': HIRES('HIRESr 0.8"', res=49000, samp=sampX, start=3900, end=8350),
                              # KCWI

                                # # # Magellan # # #
                              'MIKE 1" (r)': MIKE('MIKE 1" (r)', res=22000, samp=sampX, start=4900, end=10000),
                              'MIKE 1" (b)': MIKE('MIKE 1" (b)', res=28000, samp=sampX, start=3350, end=5000),
                              # M2FS LoRes
                              'M2FS MedRes': M2FS('M2FS MedRes', res=18000, samp=sampX, start=5132, end=5186),
                              # M2FS HiRes

                                # # # MMT # # #
                              'Hectochelle': MiscInstrument('Hectochelle', res=32000, samp=sampX, start=3800, end=9000),
                              'Hectospec': MiscInstrument('Hectochelle', res=1000, samp=sampX, start=3650, end=9200),

                                # # # VLT # # #
                              'MUSE': MiscInstrument('MUSE', res=3000, samp=sampX, start=4650, end=9300),

                                # # # JWST # # #

                              'NIRSpec G140M/F070LP': MiscInstrument('NIRSpec G140M/F070LP', res=1000, samp=sampX, start=7000, end=12700),
                              'NIRSpec G140M/F100LP': MiscInstrument('NIRSpec G140M/F100LP', res=1000, samp=sampX, start=9700, end=17999),
                              'NIRSpec G140H/F070LP': MiscInstrument('NIRSpec G140H/F070LP', res=2700, samp=sampX, start=7000, end=12700),
                              'NIRSpec G140H/F100LP': MiscInstrument('NIRSpec G140H/F100LP', res=2700, samp=sampX, start=9700, end=17999),

                                # # # MW Surveys # # #
                              'LAMOST': MiscInstrument('LAMOST', res=2000, samp=sampX, start=3900, end=9000),
                              'WEAVE': MiscInstrument('WEAVE', res=6000, samp=sampX, start=3700, end=10000),
                              'RAVE': MiscInstrument('RAVE', res=8000, samp=sampX, start=8400, end=8800),
                              'DESI (b)': MiscInstrument('DESI (b)', res=2000, samp=sampX, start=3600, end=5550),
                              'DESI (r)': MiscInstrument('DESI (r)', res=3200, samp=sampX, start=5550, end=6560),
                              'DESI (i)': MiscInstrument('DESI (i)', res=4100, samp=sampX, start=6560, end=9800),

                              'R100': MiscInstrument('R100', res=1e2, samp=sampX, start=3001, end=17999),
                              'R1000': MiscInstrument('R1000', res=1e3, samp=sampX, start=3001, end=17999),
                              'R10000': MiscInstrument('R10000', res=1e4, samp=sampX, start=3001, end=17999)
                              }
