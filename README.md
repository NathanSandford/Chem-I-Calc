# Chem-I-Calc

Chem-I-Calc is a python package for evaluating the chemical information content
of resolved star spectroscopy.
It takes advantage of the Fisher information matrix and the Cramér-Rao inequality
to quickly calculate the Cramér-Rao lower bounds (CRLBs), which give the best
theoretically achievable precision from a set of observations.

## Installation
Chem-I-Calc requires Python 3.5 or later.
It and all of its dependencies can be installed from PyPi with
```
pip install Chem-I-Calc
```
or directly from Github with
```
pip install git+https://github.com/NathanSandford/Chem-I-Calc
```

## Getting Started
### Initial file downloads
These calculations require high-resolution (R~100,000) spectra in order to calculate
spectral gradients for a wide variety of instrumental configurations. The files that
include this data are too large to host on either Github or PyPi, instead, the first time
the app looks for this data, the data is downloaded from a Google Drive, which may take
2-10 minutes depending on your internet connectivity. As chunks of the data are downloaded,
it will display in the terminal/notebook output. The label file is about 50 chunks and the
spectra file is about 26,600 chunks.

### Interactive GUI
Chem-I-Calc will eventually include a web-hosted applet for quick and easy
CRLB calculations for the broader astronomical community. The large files required
for these calculations, however, have complicated deploying this application. As an
intermediate stop-gap, the Chem-I-Calc package includes the code necessary to run the
applet locally.

If you have installed Chem-I-Calc with pip you can start up the applet by running
the command
```
run_chemicalc
```
in the python environmnent that Chem-I-Calc was installed in. If you have installed
Chem-I-Calc by cloning the repository, you will need to navigate to 
Chem-I-Calc/chemicalc_app or make sure your path includes the file run_chemicalc.

After running the command in the terminal you should see the following output:

 Serving Flask app "chemicalc_app" (lazy loading) <br>
 Environment: production <br>
 WARNING: Do not use the development server in a production environment. <br>
 Use a production WSGI server instead. <br>
 Debug mode: off <br>
 Running on http://127.0.0.1:8050/ (Press CTRL+C to quit) <br>
 
In an internet browser, navigate to the address output in the final line
(e.g., http://127.0.0.1:8050/). The app should then load and be at your disposal!

### Jupyter Notebooks
Chem-I-Calc is designed to be be both very interactive and modular, making it well suited
for use in notebook environments. This allows interested users to explore beyond the
capabilities of the GUI/applet. For example, one could easily use Chem-I-Calc with their
own stellar spectral models, include a custom Signal/Noise prescription, and otherwise
adapt the code to calculate CRLBs for their specific observations and analysis.

Tutorials for running Chem-I-Calc in a notebook can be found in Chem-I-Calc/notebooks/ and
will be expanded upon as the package grows in features.

## Authors
- Nathan Sandford (UC Berkeley, nathan_sandford@berkeley.edu)

## Collaborators
- Dan Weisz
- Yuan-Sen Ting
- Hans-Walter Rix

## Contributions
Like what this package is doing, but unsatisfied with some portion of it?
I warmly welcome any and all contributions, particularly in feature additions,
web deployment, and code optimizations. Don't hesitate to reach out to me if you have any
ideas or contributions you would like to apply.