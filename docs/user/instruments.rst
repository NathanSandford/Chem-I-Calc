.. _instruments:

Instrument Configurations
=========================

The :py:class:`InstConfig <chemicalc.reference_spectra.InstConfig>` class represents the instrumental characteristics of
the spectral observations.
Here is an overview of the class, as well as some of its important attributes and methods:

Overview
--------

.. autoclass:: chemicalc.instruments.InstConfig
    :no-members:

Attributes
----------

Wavelength
++++++++++
``InstConfig.wave``

Signal/Noise
++++++++++++
``InstConfig.snr``

Custom Wavelength Flag
++++++++++++++++++++++
``InstConfig._custom_wave``

Methods
-------

Set Custom Wavelength
+++++++++++++++++++++
.. automethod:: chemicalc.instruments.InstConfig.set_custom_wave

Reset Wavelength
++++++++++++++++
.. automethod:: chemicalc.instruments.InstConfig.reset_wave

Set Signal/Noise of Observation
+++++++++++++++++++++++++++++++
.. automethod:: chemicalc.instruments.InstConfig.set_snr

Print Summary of Instrument
+++++++++++++++++++++++++++
.. automethod:: chemicalc.instruments.InstConfig.summary

.. _pre-config_instruments:

Pre-Configured Instrument Setups
--------------------------------

To instantiate an :py:class:`InstConfig <chemicalc.reference_spectra.InstConfig>`
object from a pre-configured instrument setup:

.. code-block:: python

    from chemicalc.instruments import AllInst

    spec_config = AllInst.get_spectrograph("NAME OF CONFIGURATION")

A table summarizing the complete list of the pre-configured spectroscopic setups will be added shortly.
For now, you can see what is included by running the following code:

.. code-block:: python

    from chemicalc.instruments import AllInst

    AllInst.list_spectrographs()  # Prints summary of all pre-configured instrument setups

| Alternatively, you can look at the file:
| ``PATH_TO_INSTALLATION/Chem-I-Calc/chemicalc/data/instruments.json``
| which has the configurations organized by telescope and spectrograph.