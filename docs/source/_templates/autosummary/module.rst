{% set _parts = fullname.split('.') %}
{% set _short = (_parts[2:] | join('.')) or _parts[-1] %}
{# Public API re-exported from private submodules; autosummary's own lists skip
   these because they are imported members. See _sphinx_helpers.reexported_members. #}
{% set _reexported = reexported_members.get(fullname, {'classes': [], 'functions': []}) %}
{% set _classes = (classes + _reexported['classes']) | unique | sort | list %}
{% set _functions = (functions + _reexported['functions']) | unique | sort | list %}
{{ _short | escape | underline }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if _classes %}
Classes
-------

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in _classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if _functions %}
Functions
---------

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in _functions %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if exceptions %}
Exceptions
----------

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
