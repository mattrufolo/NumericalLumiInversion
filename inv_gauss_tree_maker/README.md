## Introduction
A beam phycisist (BP) should be as close as possible to the physics of the beam.

Parametric scans of numerical models flourished thanks to the availability 
of unprecedented computing power and the lack, the limit or complexity of the analytical tools. 

The focus of the BP should not be technicalities of the numerical simulations.
This package aims to contribute in relieve the BP from this unnecessary burden.

The "job" is the atomic unit of the simulation step and it contains all the relevant physics - it could be run on a standalone pc.
As such, **the BP should harness it**.


The BP can launch several "children" jobs almost identical to the "template" job with the exception of one or more "mutations".
This set of children jobs can be referred as a "generation".

E.g., from a given and initial "pymask" job with tune (.31,.32) - our template - we want to generate a tune scan grid of 10x10 jobs with differen tunes (a tune scan). 
This is a "generation".

Each jobs can be "parent" of other jobs thus giving birth to a second generation.

The jobs can be run in parallel, locally or remotely.

There are different phases:
- define the template job - **the most relevant**,
- cloning the template in the children jobs,
- apply one or more mutations,
- running the job.

The natural graph representing the relation between parent and children jobs is the `tree`.

The `tree` information is a "map" (e.g., a dictionary, a yaml file, a pandas dataframe) we use to define to describe the parent folder, children folders and mutations.

This package is intended to create the corresponding folders of the "generations" of the BP's study. The goal is to ease the BP's activity **while** maintaining him in close contact with the "template" job (see above, **the BP should harness it**) and offer an adequate flexibility.

## Contributors

- Philippe Belanger
- Gianni Iadarola
- Sofia Kostoglou
- Hamish Graham
- Axel Poyet
- Guido Sterbini
- Frederik Van Der Veken

## Getting started

First you need to install this package in your (virtual) environment. Presently, the suggested way is to go for local folder installation:
```
git clone https://gitlab.cern.ch/abpcomputing/sandbox/tree_maker.git
cd tree_maker
python -m pip install -e .
```

If you are on a Linux machine mounting `afs` or in the Bologna cluster you can source a pre-cooked distribution
```
source /afs/cern.ch/eng/tracking-tools/python_installations/activate_default_python
```

## Documentation

Teh package documentation can be found at https://acc-py.web.cern.ch/gitlab/abpcomputing/sandbox/tree_maker.

