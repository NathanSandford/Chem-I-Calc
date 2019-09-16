from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from mendeleev import element
from chemicalc.utils import data_dir, doppler_shift, convolve_spec, calc_MagAB, calc_gradient
from chemicalc.utils import download_package_files


precomputed_res = {'max': 300000,
                   #'high': 100000,
                   #'med': 50000,
                   #'low': 25000
                   }
precomputed_ref_id = {'max': '1I9GzorHm0KfqJ-wvZMVGbQDeyMwEu3n2'}
precomputed_cont_id = {'max': '1Fhx1KM8b6prtCGOZ3NazVeDQY-x9gOOU'}

label_file = data_dir.joinpath('reference_labels.h5')
label_id = '1-qCCjDXp2eNzRGCfIqI_2JZrzi22rFor'
if not label_file.exists():
    print('Downloading label_file')
    download_package_files(id=label_id,
                           destination=label_file)

reference_stars = list(pd.read_hdf(label_file, 'ref_list').values.flatten())
elements_included = [x.symbol for x in element(list(range(3, 100)))]
label_names = ['Teff', 'logg', 'v_micro'] + elements_included


class ReferenceSpectra:
    def __init__(self, reference: str, normalized=True, res='max', iron_scaled=False):
        self.reference = reference
        self.resolution = dict(init=precomputed_res[res])
        self.reference_file = data_dir.joinpath(f'reference_spectra_{self.resolution["init"]:06}.h5')
        if not self.reference_file.exists():
            print('Downloading reference file---this may take a few minutes but is only necessary once')
            download_package_files(id=precomputed_ref_id[res],
                                   destination=self.reference_file)
        wave_df = pd.read_hdf(self.reference_file, 'highres_wavelength')
        spec_df = pd.read_hdf(self.reference_file, reference)
        if not normalized:
            self.continuum_file = f'reference_continuum_{self.resolution["init"]:06}.h5'
            if not self.reference_file.exists():
                print('Downloading continuum file---this may take a few minutes but is only necessary once')
                download_package_files(id=precomputed_cont_id[res],
                                       destination=self.continuum_file)
            cont_df = pd.read_hdf(self.continuum_file, reference)
            spec_df *= cont_df
        label_df = pd.read_hdf(label_file, reference)
        label_df.index = label_names
        if not iron_scaled:
            label_df.loc[set(elements_included) ^ {'Fe'}] -= label_df.loc['Fe']

        self.wavelength = dict(init=wave_df.to_numpy().T[0])
        self.spectra = dict(init=spec_df.to_numpy().T)
        self.labels = label_df
        self.gradients = {}
        self.filters = {}

        self.nspectra = self.spectra['init'].shape[0]
        self.nlabels = self.labels.shape[0]

    def add_rv_spec(self, d_rv):
        self.labels.loc['RV'] = 0.0
        self.labels['fffff'] = self.labels['aaaaa']
        self.labels['ggggg'] = self.labels['aaaaa']
        self.labels.loc['RV', 'fffff'] += d_rv
        self.labels.loc['RV', 'ggggg'] -= d_rv
        tmp1 = doppler_shift(self.wavelength['init'], self.spectra['init'][0], d_rv)
        tmp2 = doppler_shift(self.wavelength['init'], self.spectra['init'][0], -d_rv)
        self.spectra['init'] = np.append(self.spectra['init'], tmp1[np.newaxis, :], axis=0)
        self.spectra['init'] = np.append(self.spectra['init'], tmp2[np.newaxis, :], axis=0)

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

    def calc_synth_phot(self, filter_set, name=None, spectrum_name='init'):
        if name is None:
            name = filter_set.name
        if spectrum_name != 'init':
            filter_interp = interp1d(filter_set.throughput.index.values, filter_set.throughput.values,
                                     axis=0, fill_value=0, bounds_error=False)
            throughput = filter_interp(self.wavelength[spectrum_name])
        else:
            throughput = filter_set.throughput.values
        self.spectra[name] = calc_MagAB(f_nu=self.spectra[spectrum_name],
                                        throughput=throughput,
                                        wave=self.wavelength[spectrum_name])
        self.wavelength[name] = np.array(list(filter_set.wave_eff.values()))
        self.filters[name] = list(filter_set.throughput.columns)
        self.resolution[name] = 0

    def calc_gradient(self, name: str, symmetric: bool = True,
                      ref_included: bool = True, v_micro_scaling: float = 1,
                      d_rv: bool = None):
        self.gradients[name] = calc_gradient(self.wavelength[name], self.spectra[name], self.labels,
                                             symmetric=symmetric, ref_included=ref_included,
                                             v_micro_scaling=v_micro_scaling, d_rv=d_rv)
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
