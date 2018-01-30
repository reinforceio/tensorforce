# -*- coding: utf-8 -*-
#
# TensorForce documentation build configuration file, created by
# sphinx-quickstart on Sun Mar 19 22:09:11 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# SPHINX_APIDOC_OPTIONS=members,undoc-members,inherited-members,show-inheritance sphinx-apidoc /data/coding/reinforce.io/tensorforce -o tensorforce

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

# import CommonMark
from recommonmark.transform import AutoStructify
from m2r import M2R


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.githubpages', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'TensorForce'
copyright = u'2017, reinforce.io'
author = u'reinforce.io'

github_doc_root = 'https://github.com/reinforceio/tensorforce/tree/master/docs/'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'0.3.3'
# The full version, including alpha/beta/rc tags.
release = u'0.3.3'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# autoclass_content = 'both'

autodoc_mock_imports = ['go_vncdriver', 'tensorflow', 'deepmind_lab', 'universe.spaces', 'gym.spaces.discrete', 'gym.wrappers',
    'mazeexp', 'ale_python_interface', 'msgpack', 'msgpack_numpy', 'cached_property',
    'tensorflow.python.training.adadelta', 'tensorflow.python.training.adagrad', 'tensorflow.python.training.adam',
    'tensorflow.python.training.gradient_descent', 'tensorflow.python.training.momentum', 'tensorflow.python.training.rmsprop',
    'tensorflow.core.util.event_pb2']

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'TensorForcedoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'TensorForce.tex', u'TensorForce Documentation',
     u'reinforce.io', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'tensorforce', u'TensorForce Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'TensorForce', u'TensorForce Documentation',
     author, 'TensorForce', 'One line description of project.',
     'Miscellaneous'),
]

m2r = M2R()
def process_docstring(app, what, name, obj, options, lines):
    """Enable markdown syntax in docstrings"""
    
    markdown = "\n".join(lines)

    # ast = cm_parser.parse(markdown)
    # html = cm_renderer.render(ast)
    rest = m2r(markdown)

    rest.replace("\r\n", "\n")
    del lines[:]
    lines.extend(rest.split("\n"))


# https://stackoverflow.com/a/5599712
def dont_skip_init(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip


def setup(app):
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: url, #  lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_eval_rst': True,
        'enable_auto_doc_ref': True,
    }, True)
    app.add_transform(AutoStructify)
    app.connect('autodoc-process-docstring', process_docstring)
    app.connect("autodoc-skip-member", dont_skip_init)
