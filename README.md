# aggregate
Some python codes to generate various aggregate particles (e.g. ballistic particle cluster, and cluster-cluster) and characterise them in a variety of ways. Note that this is not done via dynamic simulation, but takes a ray-tracing approach.

## Installation

A conda environment file is provided, so after checking out the repo you can install all dependencies with:

```conda env create```

Note that currently simulation.show() can use either matplotlib or mayavi, but the inclusion of mayavi holds back various packages (e.g. scipy).


## Usage

The `builder` package contains routines to build different aggregate types. Currently BPCA (particle-cluster) and BCCA (cluster-cluster) aggregates are implemented. Future improvements will include:

* polydisperse monomers (using a variety of distributions)
* mixed models where either single monomers or clusters can be added

The returned `simulation` object has a variety of methods to calculate aggregate properties, including:

* centre of mass
* radius of gyration
* characteristic radius
* porosity
* density
* fractal dimension (various methods)

There are also methods to display or write the simulation data to various formats:

* `show` - displays the aggregate in 3D using `mayavi` or `matplotlib`
* `projection` - calculates a 2D projection of the aggregate in a given direction
* `to_csv` - output to a simple comma separated value file
* `to_vtk' - output to a VTK file (uses [evtk](https://bitbucket.org/pauloh/pyevtk)!)
* `to_afm` - calculates a simulated AFM image assuming an infinite tip
* `to_gsf` - uses `to_afm` and outputs data to a [Gwyddion](http://gwyddion.net) simple file format
* `to_liggghts` - writes a data file that can be read by [LIGGGHTS](http://www.cfdem.com/liggghts-open-source-discrete-element-method-particle-simulation-code)

The most simple usage is as shown below:

```python
In [1]: from aggregate import builder
In [2]: agg = builder.build_bpca()
Generating particle 1024 of 1024
In [3]: agg.porosity_gyro()
Out[3]: 0.6464342819868891
In [4]: agg.com()
Out[4]: array([-0.29366738, -0.84070369,  0.81448045])
In [5]: agg.show(using='mpl')```

