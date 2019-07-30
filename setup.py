from setuptools import setup

setup(
    name='Chem-I-Calc',
    version='0.1.dev0',
    packages=['chemicalc'],
    package_data={'chemicalc': ['data/*.h5']},
    author='Nathan Sandford',
    author_email='nathan_sandford@berkeley.edu',
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
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