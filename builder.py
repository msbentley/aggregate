#!/usr/bin/python
"""
builder.py

Mark S. Bentley (mark@lunartech.org), 2016

Aggregate building module.

"""

from __future__ import print_function
from simulation import Simulation
import numpy as np


debug = False


def build_bcca(num_pcles=128, radius=1., store_aggs=False, use_stored=False, agg_path='.'):
    """
    Build a cluster-cluster agglomerate particle. This works by building two
    identical mass aggregates with m particles and allowing them to stick randomly
    to produce a 2m monomer aggregate. Two of these aggregates are then joined,
    and so on.

    Note that num_pcles must be a power of 2!

    To speed up subsequent runs, store_aggs=True will store each
    generation of aggregate to a file. If use_stored=True a random one of these
    files will be loaded. If insufficient files are available, new aggregates
    will be generated. All files are saved/loaded to/from agg_path (default=.)
    """

    num_pcles = int(num_pcles)
    if not (num_pcles != 0 and ((num_pcles & (num_pcles - 1)) == 0)):
        print('ERROR: number of particles must be a multiple of two!')
        return None

    radius = float(radius)
    if radius <= 0:
        print('ERROR: radius must be a positive value')
        return None

    num_gens = int(np.log2(num_pcles))

    # Generation files are stored as simple CSVs with the filename convention:
    # bcca_gen_<m>_<id>.csv
    # where <m> is the generation number (1=2 monomers, 2=4 monomers and so on)
    # and <id> is an incrementing ID (1=first file, etc.)

    for generation in range(num_gens):

        num_part = 2**generation

        # to build this generation, we need two of the previous generation - check
        # if we are loading these from files, and if there are enough - otherwise
        # generate anew.

        agg1 = self.build_bpca(num_pcles=2, radius=radius)
        agg1.recentre()
        agg2 = self.build_bpca(num_pcles=2, radius=radius)
        vec = random_sphere() * max(agg1.farthest() * 2.0, radius *4.)
        agg2.move(vec)

    return



def build_bpca(num_pcles=128, radius=1., overlap=None):
    """
    Build a simple ballistic particle cluster aggregate by generating particle and
     allowing it to stick where it first intersects another particle.

     If overlap= is set to a value between 0 and 1, monomers will be allowed to overlap
     by 0.5*overlap*(radius1+radius2).
     """

     # TODO: add various radius distribution options (i.e. polydisperse)

    if overlap is not None:
        if (overlap<0.) or (overlap>1.):
            print('ERROR: overlap must be either None, or 0<overlap<1')
            return None

    sim = Simulation(max_pcles=num_pcles, debug=debug)
    sim.add( (0.,0.,0.), radius)

    # generate a "proposed" particle and trajectory, and see where it intersects the
    # aggregate. add the new particle at this point!
    for n in range(num_pcles-1):

        success = False
        while not success:

            print('Generating particle %d of %d' % (n+2, num_pcles), end='\r')

            first = random_sphere() * max(sim.farthest() * 2.0, radius *4.)
            second = random_sphere() * max(sim.farthest() * 2.0, radius *4.)
            direction = (second - first)
            direction = direction/np.linalg.norm(direction)
            ids, hit = sim.intersect(first, direction, closest=True)
            if hit is None: continue

            # shift the origin along the line from the particle centre to the intersect
            new = hit + (hit-sim.pos[ids])

            # Add to the simulation, checking for overlap with existing partilces (returns False if overlap detected)
            success = sim.check(new, radius)
            if not success: continue

            # if requested, move the monomer back an amount
            if overlap is not None:
                new = hit + (hit-sim.pos[ids])*(1.-overlap)

            sim.add(new, radius)


            # if proposed particle is acceptable, add to the sim and carry on
            if success & debug: print('Adding particle at distance %f' % np.linalg.norm(hit))

    return sim


def test_random_sphere(num=1000, scale=1.):
    """
    Test routine to ensure that the random point-on-a-sphere
    generator works correctly.
    """

    points = []
    [points.append(scale*random_sphere()) for i in range(num)]
    return np.array(points)


def random_sphere():
    """
    Returns a random point on a unit sphere.
    """

    import random, math

    u = random.uniform(-1,1)
    theta = random.uniform(0,2*math.pi)
    x = math.cos(theta)*math.sqrt(1-u*u)
    y = math.sin(theta)*math.sqrt(1-u*u)
    z = u

    return np.array([x,y,z])
