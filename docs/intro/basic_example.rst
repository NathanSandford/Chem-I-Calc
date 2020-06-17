.. _basic_example:

A Basic Example
===============

Before getting started, we recommend that new Chem-I-Calc users read `Sandford et al. (In Press) <https://arxiv.org/abs/2006.08640>`_ ---
at least Sections 1-3, and 4.1 --- or our abridged summary: :ref:`Scientific/Statistical Background <background>`.

0. Install Chem-I-Calc and Prerequisites
----------------------------------------

If you have not already, please follow the instructions in :ref:`Installing Chem-I-Calc <installing>` to **both**
install the Chem-I-Calc package **and** download the requisite data files.

We find that the exploratory nature of forecasting chemical abundance precision with Chem-I-Calc
lends itself well to an interactive Python (`iPython <https://ipython.org/>`_) environment
in a `Jupyter Notebook <jupyter.org>`_. This, however, is not a requirement,
and the Chem-I-Calc package can just as easily be used in a python script if desired.

1. Initialize Observing Scenario
--------------------------------

1a. Set Reference Star
++++++++++++++++++++++

Most users will likely want to instantiate a :py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>`
object from a :ref:`pre-computed reference star <pre-computed_refstars>`
that reasonably represents the characteristics of the observed star(s).
For this example, we consider a RGB star with :math:`\log{(Z)} = -1.5` as our reference star.

.. code-block:: python

    from chemicalc import reference_spectra as ref

    RGB = ref.ReferenceSpectra(reference='RGB_m1.5')

For more information, see :ref:`Reference Stars <refstars>`.

1b. Set Spectrograph Configuration
++++++++++++++++++++++++++++++++++

There are two ways to instantiate an :py:class:`InstConfig <chemicalc.instruments.InstConfig>`.

* One of the many :ref:`pre-configured instrument setups <pre-config_instruments>` included in Chem-I-Calc can be instantiated using :py:func:`AllInst.get_spectrograph <chemicalc.instruments.AllInst.get_spectrograph>` (see :code:`d1200g` in the example below).

* A custom instrument configuration can be instantiated by calling :py:class:`InstConfig <chemicalc.instruments.InstConfig>` directly with the desired parameters (see :code:`my_spec` in the example below).

.. code-block:: python

    from chemicalc import instruments as inst

    d1200g = inst.AllInst.get_spectrograph('DEIMOS 1200G')
    my_spec = inst.InstConfig(name='My Spectrograph',
                              res=5000,    # Resolving Power
                              samp=3,      # Pixels / Resolution Element
                              start=6000,  # Blue Wavelength Bound (in Angstrom)
                              end=10000,   # Red Wavelength Bound (in Angstrom)
                              )

For more information, see :ref:`Instrument Configurations <instruments>`.

1c. Set Spectrograph Signal/Noise
+++++++++++++++++++++++++++++++++

Before we calculate the CRLBs, we must also set the Signal/Noise (S/N) of our observation using the
:py:meth:`set_snr <chemicalc.instruments.InstConfig.set_snr>` method.
This method can take the following types of arguments:

* An :code:`int` or :code:`float`: This applies a S/N that is constant with wavelength (see :code:`d1200g` below).

* A :code:`np.ndarray`: This applies a wavelength-depend S/N that is interpolated onto the wavelength grid of the instrumental configuration (see :code:`my_spec` below).

* An ETC query from :py:mod:`chemicalc.s2n` (e.g., :py:class:`Sig2NoiseDEIMOS <chemicalc.s2n.Sig2NoiseDEIMOS>`): See :ref:`ETC Queries <ETC-query>` for more details.



.. code-block:: python

    import numpy as np

    d1200g.set_snr(100)  # Set constant S/N of 100

    my_snr = np.vstack([np.linspace(my_spec.start_wavelength, my_spec.end_wavelength, 100),  # wavelength array
                        np.linspace(50, 100, 100)  # S/N array
                        ])
    my_spec.set_snr(my_snr)  # Set wavelength-dependent S/N

.. note:: Technically this does not have to occur before Steps 2 and 3,
    it just must be done before step 4 when the CRLBs are actually computed.
    In fact, if you wish to investigate the impact of the S/N on the CRLB for a given instrument,
    you only need to loop over this Step (1c) and Step 4 ---
    Steps 2 and 3 do not need to be repeated for each calculation

For more information, see :ref:`Setting the Signal/Noise <snr>`.

2. Convolve Reference Spectra to Instrument Resolution
------------------------------------------------------
Next, we convolve the high-resolution (:math:`R \sim 300000`) reference spectra down to the resolving power of our instrument setups by passing our :py:class:`InstConfig <chemicalc.instruments.InstConfig>` object to the :py:meth:`convolve <chemicalc.reference_spectra.ReferenceSpectra.convolve>` method of our :py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>` object.

.. code-block:: python

    RGB.convolve(d1200g)
    RGB.convolve(my_spec)

.. note:: If the wavelength grid of the instrument is large, this may be somewhat computationally taxing.

3. Calculate Gradient Spectra
-----------------------------
Next, we calculate the partial derivatives of the reference spectrum with respect to the stellar labels using the :py:meth:`convolve <chemicalc.reference_spectra.ReferenceSpectra.calc_gradient>` method.
This method takes as an argument either the name of an instrument setup (e.g., :code:`"DEIMOS 1200G"`) or a
:py:class:`InstConfig <chemicalc.instruments.InstConfig>` object (e.g., :code:`my_spec`).

.. code-block:: python

    RGB.calc_gradient("DEIMOS 1200G")
    RGB.calc_gradient(my_spec)

4. Calculate CRLBs
------------------
Before calculating the CRLBs, we use :py:func:`init_crlb_df <chemicalc.crlb.init_crlb_df>` to
initialize an empty :code:`pd.DataFrame` with indices corresponding to the stellar labels of
:py:class:`ReferenceSpectra <chemicalc.reference_spectra.ReferenceSpectra>`.
Then we calculate the CRLBs using :py:func:`calc_crlb <chemicalc.crlb.calc_crlb>` for each
:py:class:`InstConfig <chemicalc.instruments.InstConfig>` and store the results in a column of the CRLB DataFrame.


.. code-block:: python

    from chemiclac.crlb import init_crlb_df

    CRLB_example = init_crlb_df(RGB)

    CRLB_example['DEIMOS 1200G'] = calc_crlb(RGB, d1200g)
    CRLB_example['My Spectrograph'] = calc_crlb(RGB, my_spec)

For more information, see :ref:`Calculating CRLBs <crlb>`.

5. Apply Cutoff and Sort CRLBs
------------------------------
Using :py:func:`sort_crlb <chemicalc.crlb.sort_crlb>` we sort the DataFrame of CRLBs in order of decreasing precision and set all CRLBs above a cutoff value (here 0.3 dex) to :code:`np.nan`. Setting the argument :code:`fancy_labels=True` replaces the labels for effective temperature, surface gravity, and microturbulent velocity with LaTeX formatted labels for plotting.

.. code-block:: python

    from chemiclac.crlb import sort_crlb

    CRLB_example = sort_crlb(CRLB_example, cutoff=0.3, fancy_labels=True)

6. Plot CRLBs
-------------
Finally we can plot the CRLBs for our observing scenario!

.. code-block:: python

    from chemiclac.plot import plot_crlb

    fig = plot_crlb(CRLB_example,
                    labels='Example CRLBs\n$\log(Z)=-1.5$ RGB',
                    cutoff=0.3, cutoff_label_yoffset=0.02,
                    ylim=(0.009, 1.7))

For more information, see :ref:`Plotting with Chem-I-Calc <plotting>`.