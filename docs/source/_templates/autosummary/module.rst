{{ fullname }}
{{ underline }}

.. module:: {{fullname}}

.. automodule:: {{ fullname }}
   :noindex:

.. {% if classes %}
.. .. inheritance-diagram:: {{ fullname }}
..    :parts: 1

{% endif %}

{% block attributes %}
{% if attributes %}
.. rubric:: Lattice parameters

.. autosummary::
   :toctree:
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block functions %}
{% if functions or methods %}
.. rubric:: Functions

.. autosummary::
  :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% for item in methods %}
  {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
  :toctree:
{% for item in classes %}
  {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
  :toctree:
{% for item in classes %}
  {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
