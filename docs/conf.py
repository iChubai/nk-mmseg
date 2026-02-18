# Configuration file for the Sphinx documentation builder.
# nk-mmseg documentation - Read the Docs compatible

project = 'nk-mmseg'
copyright = '2025, DFormer-Jittor Contributors'
author = 'DFormer-Jittor Team'
release = '1.0'
version = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = None
html_favicon = None

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'jittor': ('https://cg.cs.tsinghua.edu.cn/jittor/', None),
}

nitpicky = True
