{{ fullname | escape | underline }}

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

{% if classes %}
Classes
-------

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
Functions
---------

.. currentmodule:: {{ fullname }}

.. autosummary::
   :toctree:
   :nosignatures:
{% for item in functions %}
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
