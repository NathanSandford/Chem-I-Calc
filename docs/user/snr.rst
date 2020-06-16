.. _snr:

Setting the Signal/Noise
========================

The S/N of the observation is set through the :py:meth:`InstConfig.set_snr() <chemicalc.instruments.InstConfig.set_snr>`
method:

.. automethod:: chemicalc.instruments.InstConfig.set_snr

.. _snr_constant:

Constant S/N
------------

If ``snr_input`` is an ``int``, ``float``, or ``numpy.float64``,
a constant S/N is set for all pixels in the wavelength array.

.. code-block:: python

    from chemicalc.instruments import AllInst

    d1200g_constant = AllInst.get_spectrograph("DEIMOS 1200G")
    d1200g_constant.set_snr(100)  # Set constant S/N of 100

.. _snr_array:

S/N Array
---------

If ``snr_input`` is a 2D ``numpy.ndarray``, the first row is the wavelength grid and the second row is the S/N per pixel.
The S/N is then interpolated onto the instrument's wavelength grid.

If ``snr_input`` is a 1D ``numpy.ndarray``, the wavelength grid is assumed to be linearly spaced from the instruments starting and ending wavelength.
The S/N is then interpolated onto the instrument's wavelength grid.

.. code-block:: python

    from chemicalc.instruments import AllInst

    d1200g_1Darray = AllInst.get_spectrograph("DEIMOS 1200G")
    d1200g_2Darray = AllInst.get_spectrograph("DEIMOS 1200G")

    wave_array = np.array([6500, 7000, 7500, 8000, 9000])
    snr_array = np.array([20, 30, 30, 35, 25])

    d1200g_1Darray.set_snr(snr_array)
    d1200g_2Darray.set_snr(np.vstack([wave_array, snr_array]))

.. _ETC-query:

Querying ETCs w/ Chem-I-Calc
----------------------------

Documentation Coming Soon!

WMKO
++++

MMT/Hectospec & Binospec
++++++++++++++++++

VLT
+++

MSE
+++

LCO
+++

Built In ETC Calculations
-------------------------

Documentation Coming Soon!

LBT/MODS
++++++++

Keck/FOBOS
++++++++++

TMT/WFOS
++++++++

VLT/(blue)MUSE