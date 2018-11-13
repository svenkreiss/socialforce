.. image:: https://travis-ci.org/svenkreiss/socialforce.svg?branch=master
    :target: https://travis-ci.org/svenkreiss/socialforce


Social Force Model
==================

.. code-block::

    citation coming soon ...


Install and Run
===============

.. code-block:: sh

    # install from PyPI
    pip install 'socialforce[test,plot]'

    # or install from source
    pip install -e '.[test,plot]'

    # run linting and tests
    pylint socialforce
    pytest tests/*.py

    # plots of simulations
    pytest -s tests/scenarios_*.py

    # run fits and create plots
    pytest -s tests/fit_pedped.py::test_opposing_mlp
    pytest -s tests/fit_pedped.py::test_circle_mlp


Ped-Ped-Space Scenarios
=======================

+----------------------------------------+----------------------------------------+
| .. image:: docs/separator.gif          | .. image:: docs/gate.gif               |
+----------------------------------------+----------------------------------------+
| Emergent lane formation with           | Emergent lane formation with           |
| 30 pedestrians:                        | 60 pedestrians:                        |
|                                        |                                        |
| .. image:: docs/walkway_30.gif         | .. image:: docs/walkway_60.gif         |
+----------------------------------------+----------------------------------------+


Ped-Ped Scenarios
=================

+----------------------------------------+----------------------------------------+
| .. image:: docs/crossing.png           | .. image:: docs/narrow_crossing.png    |
+----------------------------------------+----------------------------------------+
| .. image:: docs/opposing.png           | .. image:: docs/2opposing.png          |
+----------------------------------------+----------------------------------------+
