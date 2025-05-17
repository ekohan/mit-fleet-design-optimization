import datetime
import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

# -- Project information -----------------------------------------------------
project = 'MIT Fleet Design Optimisation'
author = 'MIT Capstone Team'
copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- HTML --------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme' 