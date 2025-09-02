from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='MOBSTERm',
      version='0.1.3',
      description='Pyro model to perform multivariate subclonal deconvolution',
      url='https://github.com/caravagnalab/MOBSTERm',
      author='Elena Rivaroli',
      author_email='elenarivaroli@gmail.com',
      license="GPL-3.0",
      packages=['MOBSTERm'],
      python_requires=">=3.11",
      install_requires=[
            'matplotlib>=3.10',
            'pandas>=2.2.2',
            'pyro-ppl>=1.9.1',
            'numpy>=1.26.4',
            'scikit-learn>=1.5',
            'torch>=2.3.0',
            'seaborn>=0.13.2',
            'scipy>=1.14.1'
      ]
)