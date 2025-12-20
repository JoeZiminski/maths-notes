# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = "maths-notes"
copyright = "2025, joe ziminski"
author = "joe ziminski"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_design",
]
myst_enable_extensions = [
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = []

# Allow both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_title = "Maths Notes"

html_static_path = ["_static"]

html_theme_options = {
    "navbar_align": "content",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JoeZiminski/maths-notes",
            "icon": "fa-brands fa-github",
        }
    ],
}

html_sidebars = {
    "**": ["sidebar-nav-bs.html"]
}

# -- nbsphinx options --------------------------------------------------------

nbsphinx_execute = "always"   # don't re-run notebooks on build
