from setuptools import setup

setup(
    name='Chem-I-Calc',
    version='0.1',
    packages=['chemicalc', 'chemicalc_app'],
    scripts=['chemicalc_app/run_chemicalc'],
    author='Nathan Sandford',
    author_email='nathan_sandford@berkeley.edu',
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        'tqdm',
        'requests',
        'dash',
        'flask',
        'mendeleev',
        'matplotlib',
        'plotly',
        'scipy',
        'mechanicalsoup',
        'pandas',
        'numpy',
        'tables',
    ],
)