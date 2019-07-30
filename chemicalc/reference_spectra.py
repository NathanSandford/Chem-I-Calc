import pandas as pd
from mendeleev import element
from chemicalc.utils import data_dir, convolve_spec, calc_gradient

precomputed_res = {'max': 300000,
                   'high': 100000,
                   'med': 50000,
                   'low': 25000}

reference_stars = list(pd.read_hdf(data_dir + 'reference_spectra_300000.h5', 'ref_list').values.flatten())
elements_included = [x.symbol for x in element(list(range(3, 100)))]
label_names = ['Teff', 'logg', 'v_micro'] + elements_included


class ReferenceSpectra:
    def __init__(self, reference: str, res='max', iron_scaled=False):
        self.reference = reference
        self.resolution = dict(init=precomputed_res[res])
        self.reference_file = f'reference_spectra_{self.resolution["init"]:06}.h5'
        wave_df = pd.read_hdf(data_dir + self.reference_file, 'highres_wavelength')
        spec_df = pd.read_hdf(data_dir + self.reference_file, reference)
        label_df = pd.read_hdf(data_dir + 'reference_labels.h5', reference)
        label_df.index = label_names
        if not iron_scaled:
            label_df.loc[set(elements_included) ^ {'Fe'}] -= label_df.loc['Fe']

        self.wavelength = dict(init=wave_df.to_numpy().T[0])
        self.spectra = dict(init=spec_df.to_numpy().T)
        self.labels = label_df
        self.gradients = {}

        self.nspectra = self.spectra['init'].shape[0]
        self.nlabels = self.labels.shape[0]

    def convolve(self, instrument, name=None):
        if name is None:
            name = instrument.name
        outwave = instrument.get_wave()
        self.spectra[name] = convolve_spec(wave=self.wavelength['init'],
                                           spec=self.spectra['init'],
                                           resolution=instrument.R_res,
                                           outwave=outwave,
                                           res_in=self.resolution['init'])
        self.wavelength[name] = outwave
        self.resolution[name] = instrument.R_res

    def calc_gradient(self, name: str, symmetric: bool = True,
                      ref_included: bool = True, v_micro_scaling: float = 1e5):
        self.gradients[name] = calc_gradient(self.spectra[name], self.labels,
                                             symmetric=symmetric, ref_included=ref_included,
                                             v_micro_scaling=v_micro_scaling)
        self.gradients[name].columns = self.wavelength[name]

    def zero_gradients(self, name: str, labels: list):
        self.gradients[name].loc[labels] = 0

    def get_names(self):
        return list(self.spectra.keys())

    def reset(self):
        init_resolution = self.resolution['init']
        self.resolution.clear()
        self.resolution['init'] = init_resolution

        init_wavelength = self.wavelength['init']
        self.wavelength.clear()
        self.wavelength['init'] = init_wavelength

        init_spectra = self.spectra['init']
        self.spectra.clear()
        self.spectra['init'] = init_spectra

        del self.gradients
