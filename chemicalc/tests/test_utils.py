import pytest

import numpy as np
import chemicalc.reference_spectra as ref
from chemicalc import utils as u


def test_alpha_el():
    assert isinstance(u.alpha_el, list)
    assert u.alpha_el == ref.alpha_el
    for el in u.alpha_el:
        assert el in ref.elements_included


def test_find_nearest():
    test_array = np.arange(100)
    test_list = list(range(100))
    assert u.find_nearest(test_array, 50) == 50
    assert u.find_nearest(test_list, 50) == 50
    with pytest.raises(TypeError):
        u.find_nearest("str")
        u.find_nearest(range(100))


def test_find_nearest_idx():
    test_array = np.arange(100)
    test_list = list(range(100))
    assert u.find_nearest_idx(test_array, 50) == int(50)
    assert u.find_nearest_idx(test_list, 50) == int(50)
    with pytest.raises(TypeError):
        u.find_nearest_idx("str")
        u.find_nearest_idx(range(100))


def test_gen_wave_template():
    wave = u.generate_wavelength_template(6500, 9000, 6500, 4, False)
    log_wave_diff = np.diff(np.log10(wave))
    assert wave.shape == (8463,)
    assert np.all(u.generate_wavelength_template(6500, 9000, 6500, 4, True) == wave[:-1])
    assert np.std(log_wave_diff) < 1e-10
    with pytest.raises(TypeError):
        u.generate_wavelength_template('str', 6000, 2000, 3, False)
        u.generate_wavelength_template(5000, 'str', 2000, 3, False)
        u.generate_wavelength_template(5000, 6000, 'str', 3, False)
        u.generate_wavelength_template(5000, 6000, 2000, 'str', False)
    with pytest.raises(ValueError):
        u.generate_wavelength_template(-1, 6000, 2000, 3, False)
        u.generate_wavelength_template(5000, 6000, -1, 3, False)
        u.generate_wavelength_template(5000, 6000, 2000, -1, False)
        u.generate_wavelength_template(6000, 5000, 2000, 3, False)


def test_convolve_spec():
    pass


def test_calc_grad():
    pass


def test_calc_crlb():
    pass


def test_sort_crlb():
    pass


def test_kpc_to_mu():
    ten_pc_in_kpc = 10 * 1e-3
    hund_pc_in_kpc = 100 * 1e-3
    assert u.kpc_to_mu(ten_pc_in_kpc) == 0
    assert u.kpc_to_mu(hund_pc_in_kpc) == 5
    assert u.kpc_to_mu(0) == -np.inf
    assert np.all(u.kpc_to_mu(ten_pc_in_kpc * np.ones(10)) == np.zeros(10))
    assert np.all(u.kpc_to_mu([ten_pc_in_kpc] * 10) == np.zeros(10))
    with pytest.raises(TypeError):
        u.kpc_to_mu("str")


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
