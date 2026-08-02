# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Point to the root directory where your .py files or source package live
# If conf.py is in 'docs/source/', '../../' goes up two levels to the root
sys.path.insert(0, os.path.abspath('../../'))

project = 'MOBSTERm'
copyright = '2026, Elena Rivaroli'
author = 'Elena Rivaroli'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = [
    "pyro",
    "seaborn",
    "scipy",
    "torch",
    "numpy",
    "sklearn",
    "pandas",
    "rich",
    "matplotlib"
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_sidebars = {
    '**': []
}
