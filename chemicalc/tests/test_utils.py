import pytest
import os
from pathlib import Path

import numpy as np
import chemicalc.reference_spectra as ref
from chemicalc import utils as u

test_file_dir = Path(os.path.dirname(__file__)).joinpath("files")

# ToDo: Clean up with @pytest.mark.parametrize()


def test_alpha_el():
    assert isinstance(u.alpha_el, list)
    assert u.alpha_el == ref.alpha_el
    for el in u.alpha_el:
        assert el in ref.elements_included


def test_find_nearest():
    test_array = np.arange(100)
    test_list = list(range(100))
    assert u.find_nearest_val(test_array, 50) == 50
    assert u.find_nearest_val(test_list, 50) == 50
    with pytest.raises(TypeError):
        u.find_nearest_val("str", 50)
        u.find_nearest_val(range(100), 50)
        u.find_nearest_val(test_array, "str")


def test_find_nearest_idx():
    test_array = np.arange(100)
    test_list = list(range(100))
    assert u.find_nearest_idx(test_array, 50) == int(50)
    assert u.find_nearest_idx(test_list, 50) == int(50)
    with pytest.raises(TypeError):
        u.find_nearest_idx("str", 50)
        u.find_nearest_idx(range(100), 50)
        u.find_nearest_idx(test_array, "str")


def test_gen_wave_template():
    wave = u.generate_wavelength_template(6500, 9000, 6500, 4, False)
    log_wave_diff = np.diff(np.log10(wave))
    assert wave.shape == (8463,)
    assert np.all(np.load(test_file_dir.joinpath("wave.npy")) == wave)
    assert np.all(
        u.generate_wavelength_template(6500, 9000, 6500, 4, True) == wave[:-1]
    )
    assert np.std(log_wave_diff) < 1e-10
    with pytest.raises(TypeError):
        u.generate_wavelength_template("str", 6000, 2000, 3, False)
        u.generate_wavelength_template(5000, "str", 2000, 3, False)
        u.generate_wavelength_template(5000, 6000, "str", 3, False)
        u.generate_wavelength_template(5000, 6000, 2000, "str", False)
    with pytest.raises(ValueError):
        u.generate_wavelength_template(-1, 6000, 2000, 3, False)
        u.generate_wavelength_template(5000, 6000, -1, 3, False)
        u.generate_wavelength_template(5000, 6000, 2000, -1, False)
        u.generate_wavelength_template(6000, 5000, 2000, 3, False)


def test_convolve_spec():
    star = ref.ReferenceSpectra(reference="RGB_m1.5", alpha_included=True)
    wave = star.wavelength["init"]
    spec = star.spectra["init"]
    outwave = np.load(test_file_dir.joinpath("wave.npy"))
    res_in = star.resolution["init"]
    convolved_spec = u.convolve_spec(
        wave=wave, spec=spec[0], resolution=6500, outwave=outwave, res_in=res_in,
    )
    assert convolved_spec.ndim == spec[0].ndim
    assert convolved_spec.shape[0] == outwave.shape[0]
    convolved_specs = u.convolve_spec(
        wave=wave, spec=spec, resolution=6500, outwave=outwave, res_in=res_in,
    )
    assert convolved_specs.ndim == spec.ndim
    assert convolved_specs.shape[0] == spec.shape[0]
    assert convolved_specs.shape[1] == outwave.shape[0]
    assert np.all(np.abs(convolved_specs[0] - convolved_spec) < 1e-10)
    assert np.all(
        np.load(test_file_dir.joinpath("convolved_spec.npy")) == convolved_specs
    )
    with pytest.raises(TypeError):
        u.convolve_spec(
            wave=100, spec=spec, resolution=6500, outwave=outwave, res_in=res_in
        )
        u.convolve_spec(
            wave=wave, spec=100, resolution=6500, outwave=outwave, res_in=res_in
        )
        u.convolve_spec(
            wave=wave, spec=spec, resolution="str", outwave=outwave, res_in=res_in
        )
        u.convolve_spec(
            wave=wave, spec=spec, resolution=6500, outwave=100, res_in=res_in
        )
        u.convolve_spec(
            wave=wave, spec=spec, resolution=6500, outwave=outwave, res_in="str"
        )
    with pytest.raises(ValueError):
        u.convolve_spec(
            wave=wave, spec=spec, resolution=10000, outwave=outwave, res_in=5000
        )
        u.convolve_spec(
            wave=wave[:-1], spec=spec, resolution=6500, outwave=outwave, res_in=res_in,
        )
        u.convolve_spec(
            wave=wave[:-1],
            spec=spec[0],
            resolution=6500,
            outwave=outwave,
            res_in=res_in,
        )
        u.convolve_spec(
            wave=wave,
            spec=spec,
            resolution=6500,
            outwave=np.random.permutation(wave),
            res_in=res_in,
        )
        u.convolve_spec(
            wave=wave,
            spec=spec,
            resolution=6500,
            outwave=np.random.permutation(outwave),
            res_in=res_in,
        )
        u.convolve_spec(
            wave=wave,
            spec=spec[:, np.newaxis],
            resolution=6500,
            outwave=outwave,
            res_in=res_in,
        )


def test_doppler_shift():
    star = ref.ReferenceSpectra(reference="RGB_m1.5")
    wave = star.wavelength["init"]
    spec = star.spectra["init"][0]
    shifted_spec = u.doppler_shift(wave, spec, 10)
    assert shifted_spec.shape == spec.shape
    # ToDo: UnitTests
    #assert np.all(
    #    np.load(test_file_dir.joinpath("doppler_spec.npy")) == shifted_spec
    #)
    with pytest.raises(TypeError):
        u.doppler_shift('str', spec, 10)
        u.doppler_shift(wave, 'str', 10)
        u.doppler_shift(wave, spec, 'str')
    with pytest.raises(ValueError):
        u.doppler_shift(wave, spec, -10)
        u.doppler_shift(np.random.permutation(wave), spec, 10)


def test_calc_grad():
    star = ref.ReferenceSpectra(reference="RGB_m1.5", alpha_included=True)
    spec = star.spectra["init"]
    labels = star.labels
    sym_grad = u.calc_gradient(spectra=spec, labels=labels, symmetric=True, ref_included=True)
    sym_grad_noref = u.calc_gradient(spectra=spec[1:], labels=labels.iloc[:, 1:], symmetric=True, ref_included=False)
    asym_grad = u.calc_gradient(spectra=spec, labels=labels, symmetric=False, ref_included=True)
    # ToDo: UnitTests
    with pytest.raises(TypeError):
        u.calc_gradient(spectra='str', labels=labels, symmetric=True, ref_included=True)
        u.calc_gradient(spectra=spec, labels='str', symmetric=True, ref_included=True)
    with pytest.raises(ValueError):
        # Ref included, but ref_included=False
        u.calc_gradient(spectra=spec, labels=labels, symmetric=True, ref_included=False)
        u.calc_gradient(spectra=spec, labels=labels, symmetric=False, ref_included=False)
        # symmetric gradient w/o necessary spectra
        u.calc_gradient(spectra=spec[::2], labels=labels.iloc[:,::2], symmetric=True, ref_included=True)
        u.calc_gradient(spectra=spec[::2], labels=labels.iloc[:, ::2], symmetric=True, ref_included=False)
        u.calc_gradient(spectra=spec[::2], labels=labels.iloc[:, ::2], symmetric=False, ref_included=False)
        # Ref not included, but ref_included=True
        u.calc_gradient(spectra=spec[1:], labels=labels.iloc[:, 1:], symmetric=True, ref_included=True)
        u.calc_gradient(spectra=spec[1:], labels=labels.iloc[:, 1:], symmetric=False, ref_included=True)
        u.calc_gradient(spectra=spec[1::2], labels=labels.iloc[:, 1::2], symmetric=True, ref_included=True)
        u.calc_gradient(spectra=spec[1::2], labels=labels.iloc[:, 1::2], symmetric=False, ref_included=True)
        # Ref not included, but asymmetric gradient attempted
        u.calc_gradient(spectra=spec[1:], labels=labels.iloc[:, 1:], symmetric=False, ref_included=False)


def test_kpc_to_mu():
    ten_pc_in_kpc = 10 * 1e-3
    hund_pc_in_kpc = 100 * 1e-3
    assert u.kpc_to_mu(ten_pc_in_kpc) == 0
    assert u.kpc_to_mu(hund_pc_in_kpc) == 5
    assert np.all(u.kpc_to_mu(ten_pc_in_kpc * np.ones(10)) == np.zeros(10))
    assert np.all(u.kpc_to_mu([ten_pc_in_kpc] * 10) == np.zeros(10))
    with pytest.raises(TypeError):
        u.kpc_to_mu("str")
    with pytest.raises(ValueError):
        u.kpc_to_mu(-1)


def test_mu_to_kpc():
    ten_pc_in_kpc = 10 * 1e-3
    assert u.mu_to_kpc(u.kpc_to_mu(ten_pc_in_kpc)) == ten_pc_in_kpc
    assert np.all(
        u.mu_to_kpc(u.kpc_to_mu(ten_pc_in_kpc * np.ones(10)))
        == ten_pc_in_kpc * np.ones(10)
    )
    assert np.all(
        u.mu_to_kpc(u.kpc_to_mu([ten_pc_in_kpc] * 10)) == ten_pc_in_kpc * np.ones(10)
    )
    with pytest.raises(TypeError):
        u.mu_to_kpc("str")
