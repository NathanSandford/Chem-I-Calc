import os
from pathlib import Path
import numpy as np
from chemicalc.instruments import AllInstruments

spectrographs = AllInstruments().spectrographs

tmp_data_dir = Path(os.path.dirname(__file__)).joinpath('/tmp/')
tmp_data_dir.mkdir(exist_ok=True)


def load_preset_specs(inst_name):
    preset = spectrographs[inst_name]
    inst = dict(
        name=inst_name,
        start_wavelength=preset.start_wavelength,
        end_wavelength=preset.end_wavelength,
        R_res=preset.R_res,
        R_samp=preset.R_samp
    )
    return inst


lab_tab_text = np.array(["Teff", "g", "v_micro",
                         "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                         "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
                         "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
                         "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
                         "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                         "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
                         "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                         "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
                         "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
                         "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
                         "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf"])


prior_tooltips = [u'1-\u03C3 prior on effective temperature in K. '
                  u'Leave empty to include no prior information.'
                  u' Set to 0 to fix value.',
                  u'1-\u03C3 prior on log(g) in dex.'
                  u' Leave empty to include no prior information.'
                  u' Set to 0 to fix value.',
                  u'1-\u03C3 prior on [Fe/H] in dex.'
                  u' Leave empty to include no prior information.'
                  u' Set to 0 to fix value.']


sample_labels = ['T<sub>eff</sub> (100 K)', 'log(g)', 'v<sub>micro</sub> (km/s)',
                 "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                 "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K"
                 ]

snr_options = ['Constant', 'From ETC']
etc_options = ['WMKO']