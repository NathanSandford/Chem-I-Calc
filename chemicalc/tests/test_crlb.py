import pytest
import os
from pathlib import Path

import chemicalc.reference_spectra as ref
from chemicalc import crlb

test_file_dir = Path(os.path.dirname(__file__)).joinpath("files")

# ToDo: Clean up with @pytest.mark.parametrize()


@pytest.mark.xfail(reason="Test not implemented fully")
def test_calc_crlb():
    star = ref.ReferenceSpectra(reference="RGB_m1.5")
    test_inst = None  # Will implement soon
    missing_inst = None  # Will implement soon
    # ToDo: Unit Tests
    with pytest.raises(TypeError):
        crlb.calc_crlb('str', test_inst, None, False, False, 10000)
        crlb.calc_crlb(star, 'str', None, False, False, 10000)
        crlb.calc_crlb(star, ['str'], None, False, False, 10000)
        crlb.calc_crlb(star, test_inst, 'str', False, False, 10000)
        crlb.calc_crlb(star, test_inst, {'non-existent label': 'str'}, False, False, 10000)
        crlb.calc_crlb(star, test_inst, None, False, False, 'str')
    with pytest.raises(KeyError):
        # Missing Instrument
        crlb.calc_crlb(star, missing_inst, None, False, False, 10000)
        # Nonexistent Prior Label
        crlb.calc_crlb(star, test_inst, {'label': 1.0}, False, False, 10000)
    with pytest.raises(ValueError):
        # Missing Alpha
        crlb.calc_crlb(star, test_inst, None, True, False, 10000)


@pytest.mark.xfail(reason="Test not implemented fully")
def test_sort_crlb():
    # ToDo: Unit Tests
    raw_crlb = None  # Will implement soon
    with pytest.raises(TypeError):
        crlb.sort_crlb('str', 0.3, "inst")
        crlb.sort_crlb(crlb, 'str', "inst")
        crlb.sort_crlb(crlb, 0.3, 100)
    with pytest.raises(KeyError):
        crlb.sort_crlb(crlb, 0.3, "non-existent instrument")