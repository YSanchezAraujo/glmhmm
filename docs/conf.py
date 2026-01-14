
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'glmhmm'
copyright = '2025, User'
author = 'User'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']

autodoc_mock_imports = ["jax", "numpy", "matplotlib", "seaborn", "optax"]
