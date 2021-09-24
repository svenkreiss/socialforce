[![Build Status](https://travis-ci.org/svenkreiss/socialforce.svg?branch=master)](https://travis-ci.org/svenkreiss/socialforce)
[Executable Book](https://www.svenkreiss.com/socialforce/)

# Deep Social Force

> __Deep Social Force__<br />
> _[Sven Kreiss](https://www.svenkreiss.com)_, 2021.
>
> The Social Force model introduced by Helbing and Molnar in 1995
> is a cornerstone of pedestrian simulation. This paper
> introduces a differentiable simulation of the Social Force model
> where the assumptions on the shapes of interaction potentials are relaxed
> with the use of universal function approximators in the form of neural
> networks.
> Classical force-based pedestrian simulations suffer from unnatural
> locking behavior on head-on collision paths. In addition, they cannot
> model the bias
> of pedestrians to avoid each other on the right or left depending on
> the geographic region.
> My experiments with more general interaction potentials show that
> potentials with a sharp tip in the front avoid
> locking. In addition, asymmetric interaction potentials lead to a left or right
> bias when pedestrians avoid each other.


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


# Download TrajNet++ Data

The [Executable Book](https://www.svenkreiss.com/socialforce/)
requires some real-world data for the TrajNet++ section.
This is how to download and unzip it to the right folder:

```
wget -q https://github.com/vita-epfl/trajnetplusplusdata/releases/download/v4.0/train.zip
mkdir data-trajnet
unzip train.zip -d data-trajnet
```