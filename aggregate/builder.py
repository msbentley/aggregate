#!/usr/bin/python
"""
builder.py

Mark S. Bentley (mark@lunartech.org), 2016

Aggregate building module. This is the driver for building aggregates of
spherical and spheroidal aggregates using different aggregation techniques.

Currently available types are:

    build_bcca() - Ballistic cluster-cluster aggregate
    build_bpca() - Ballistic particle cluster aggregate

"""

from __future__ import print_function
from simulation import Simulation
import numpy as np


debug = False


# TODO: look at a "sphere cylinder" intersection to see if there is an easy way (akin
# to the bounding box method for spheres) of checking for aggregate intersect and
# discarding those that can't possibly hit!

# TODO: in general consider spheroidal monomers instead of spheres

# TODO: set up a logger here, instead of global debug/prints

# TODO: Add a builder that mixes cluster and particle aggregates with some distribution

# TODO: general - add polydisperse monomers in a single aggregate

# TODO: add QBCCA? (http://iopscience.iop.org/article/10.1088/0004-637X/707/2/1247/pdf)

# TODO: add BAM1/BAM2 type aggregation? (allow for rolling rather than hit + stick)

def build_hierarchical():

    pass



def build_bcca(num_pcles=1024, radius=0.5, overlap=None, store_aggs=False, use_stored=False, agg_path='.', constrain_dir=True):
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

    import glob, os

    num_pcles = int(num_pcles)
    if not (num_pcles != 0 and ((num_pcles & (num_pcles - 1)) == 0)):
        print('ERROR: number of particles must be a multiple of two!')
        return None

    radius = float(radius)
    if radius <= 0:
        print('ERROR: radius must be a positive value')
        return None

    if overlap is not None:
        if (overlap<0.) or (overlap>1.):
            print('ERROR: overlap must be either None, or 0<overlap<1')
            return None

    num_gens = int(np.log2(num_pcles))

    # Generation files are stored as simple CSVs with the filename convention:
    # bcca_gen_<m>_<id>.csv
    # where <m> is the generation number (1=2 monomers, 2=4 monomers and so on)
    # and <id> is an incrementing ID (1=first file, etc.)

    # first run, generate 2 monomer BPCA aggregates
    agg_list = []
    [agg_list.append(build_bpca(num_pcles=2, radius=radius, output=False, overlap=overlap)) for i in range(num_pcles/2)]
    [agg.recentre() for agg in agg_list]

    # loop over generations needed
    for idx, gen in enumerate(range(num_gens-1,0,-1)):

        num_aggs = 2**gen
        print('INFO: Building generation %d with %d aggregates of %d monomers' % (idx+1,num_aggs,2**(idx+1)))

        next_list = [] # the list of next generation aggregate (half as big as agg_list)
        
        for agg_idx in range(0,num_aggs,2):
            sim = Simulation(max_pcles=num_pcles)
            agg1 = agg_list[agg_idx]
            agg2 = agg_list[agg_idx+1]
            sim.add_agg(agg1)
            # TODO - calculate the optimum value instead of 10 here!
            vec = random_sphere() * max(sim.farthest() * 10.0, radius *4.)
            agg2.move(vec)

            success = False
            while not success:

                second = random_sphere() * max(agg1.farthest() * 10.0, radius *4.)

                if constrain_dir:
                    direction = (second - vec)
                else:
                    direction = second + random_sphere()

                direction = direction/np.linalg.norm(direction)
                ids, dist, hit = sim.intersect(agg2.pos, direction, closest=True)

                if hit is None:
                    continue
                else:
                    agg2.move(direction*dist)

                    # now need to shift to avoid any overlap - query the intersect between
                    # two monomers that will be colliding
                    agg2.move(hit-sim.pos[np.where(sim.id==ids)[0][0]])

                    # check if there are any overlaps in the domain
                    success = sim.check(agg2.pos, agg2.radius)
                    if not success: continue

                    # if requested, move the monomer back an amount
                    if overlap is not None:
                        agg2.move( (sim.pos[np.where(sim.id==ids)[0][0]]-hit)*(overlap) )

                    sim.add_agg(agg2)
                    sim.recentre()
                    next_list.append(sim)

                    if store_aggs:
                        # bcca_gen_<m>_<id>.csv
                        agg_files = glob.glob(os.path.join(agg_path, 'bcca_gen_%03d_*.csv' % (idx+1)))
                        id_list = [int(os.path.basename(f).split('_')[3].split('.')[0]) for f in agg_files]
                        agg_id = 1 if len(id_list) == 0 else max(id_list) + 1
                        agg_file = os.path.join(agg_path, 'bcca_gen_%03d_%03d.csv' % (idx+1, agg_id))
                        agg2.to_csv(agg_file)

        agg_list = next_list

    return next_list[0]



def build_bpca(num_pcles=1024, radius=0.5, overlap=None, output=True):
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

            if output: print('Generating particle %d of %d' % (n+2, num_pcles), end='\r')

            first = random_sphere() * max(sim.farthest() * 2.0, radius *4.)
            second = random_sphere() * max(sim.farthest() * 2.0, radius *4.)
            direction = (second - first)
            direction = direction/np.linalg.norm(direction)
            ids, hit = sim.intersect(first, direction, closest=True)
            if hit is None: continue

            # shift the origin along the line from the particle centre to the intersect
            new = hit + (hit-sim.pos[np.where(sim.id==ids)[0][0]])

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
