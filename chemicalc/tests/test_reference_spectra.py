import pytest

import chemicalc.reference_spectra as ref
from chemicalc import utils as u

# ToDo: Clean up with @pytest.mark.parametrize()


def test_alpha_el():
    assert isinstance(ref.alpha_el, list)
    assert ref.alpha_el == u.alpha_el
    for el in ref.alpha_el:
        assert el in ref.elements_included


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_init():
    # ToDo: Implement Tests
    star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_addrv():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_convolve():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_calcgrad():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_zerograd():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_getnames():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass


@pytest.mark.skip(reason="Test not implemented")
def test_RefSpec_reset():
    # ToDo: Implement Tests
    #star = ref.ReferenceSpectra(reference="RGB_m1.5")
    pass
