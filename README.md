[![Build Status](https://travis-ci.org/svenkreiss/socialforce.svg?branch=master)](https://travis-ci.org/svenkreiss/socialforce)


# Social Force Model

```
    citation coming soon ...
```


# Install and Run

```sh
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
```


# Ped-Ped-Space Scenarios

<img src="docs/separator.gif" height=200 />
<img src="docs/gate.gif" height=200 />

Emergent lane forming behavior with 30 and 60 pedestrians:

<img src="docs/walkway_30.gif" height=200 />
<img src="docs/walkway_60.gif" height=200 />


# Ped-Ped Scenarios

Crossing:

<img src="docs/crossing.png" height=200 />
<img src="docs/narrow_crossing.png" height=200 />

Opposing:

<img src="docs/opposing.png" height=200 />
<img src="docs/2opposing.png" height=200 />
