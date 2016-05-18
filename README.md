# aggregate
Some python codes to generate various aggregate particles (e.g. ballistic particle cluster, and cluster-cluster). Note that this package is in a very early stage of development!

## Installation

A conda environment file is provided, so after checking out the repo you can install all dependencies with:

```conda env create```

Note that currently simulation.show() can use either matplotlib or mayavi, but the inclusion of mayavi holds back various packages (e.g. scipy).


## Usage

The `builder` package contains routines to build different aggregate types. Currently only BPCA is implemented using a ray-tracing routine to propose new monomer positions, which are then checked for overlap before being inserted into the simulation.

The returned `simulation` object has a variety of methods to calculate aggregate properties, including:

* centre of mass
* radius of gyration
* characteristic radius
* porosity
* fractal dimension


