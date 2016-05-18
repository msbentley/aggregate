#!/usr/bin/python
"""
builder.py

Mark S. Bentley (mark@lunartech.org), 2016

Aggregate building module.

"""

from __future__ import print_function
import numpy as np
from particle import Particle
from simulation import Simulation

debug = False


def build_bpca(num_pcles=128, radius=1.):
    """Build a simple ballistic particle cluster aggregate by generating particle and
     allowing it to stick where it first intersects another particle."""

    sim = Simulation(max_pcles=num_pcles)
    sim.add(Particle(x=0., y=0., z=0., r=radius), check=False)

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
            hit += (hit-sim.pos[ids])
            # Add to the simulation, checking for overlap with existing partilces (returns False if overlap detected)
            success = sim.add(Particle(x=hit[0], y=hit[1], z=hit[2], r=radius), check=True)
            # if proposed particle is acceptable, add to the sim and carry on
            if success & debug: print('Adding particle at distance %f' % np.linalg.norm(hit))

    return sim


def test_random_sphere(num=1000, scale=1.):
    """Test routine to ensure that the random point-on-a-sphere
    generator works correctly"""

    points = []
    [points.append(scale*random_sphere()) for i in range(num)]
    return np.array(points)


def random_sphere():
    """Returns a random point on a unit sphere"""

    import random, math

    u = random.uniform(-1,1)
    theta = random.uniform(0,2*math.pi)
    x = math.cos(theta)*math.sqrt(1-u*u)
    y = math.sin(theta)*math.sqrt(1-u*u)
    z = u

    return np.array([x,y,z])





def sphere_line(centre, radius, line_start, line_end):
    """Checks for line-sphere intersection (zero, one or two). Returns
    coordinates of the intersects, or None for each of the two possible
    points"""

    def square(f):
        return f * f
    from math import sqrt

    p1 = p2 = None

    a = square(line_end[0] - line_start[0]) + square(line_end[1] - line_start[1]) + square(line_end[2] - line_start[2])

    b = 2.0 * ((line_end[0] - line_start[0]) * (line_start[0] - centre[0]) +
               (line_end[1] - line_start[1]) * (line_start[1] - centre[1]) +
               (line_end[2] - line_start[2]) * (line_start[2] - centre[2]))

    c = (square(centre[0]) + square(centre[1]) + square(centre[2]) + square(line_start[0]) +
            square(line_start[1]) + square(line_start[2]) -
            2.0 * (centre[0] * line_start[0] + centre[1] * line_start[1] + centre[2] * line_start[2]) - square(radius))

    i = b * b - 4.0 * a * c

    if i < 0.0:
        pass  # no intersections
    elif i == 0.0:
        # one intersection
        p[0] = 1.0

        mu = -b / (2.0 * a)
        p1 = (line_start[0] + mu * (line_end[0] - line_start[0]),
              line_start[1] + mu * (line_end[1] - line_start[1]),
              line_start[2] + mu * (line_end[2] - line_start[2]),
              )

    elif i > 0.0:
        # first intersection
        mu = (-b + sqrt(i)) / (2.0 * a)
        p1 = (line_start[0] + mu * (line_end[0] - line_start[0]),
              line_start[1] + mu * (line_end[1] - line_start[1]),
              line_start[2] + mu * (line_end[2] - line_start[2]),
              )

        # second intersection
        mu = (-b - sqrt(i)) / (2.0 * a)
        p2 = (line_start[0] + mu * (line_end[0] - line_start[0]),
              line_start[1] + mu * (line_end[1] - line_start[1]),
              line_start[2] + mu * (line_end[2] - line_start[2]),
              )

    p1 = np.array(p1) if p1 is not None else p1
    p2 = np.array(p2) if p2 is not None else p2

    return p1, p2




#=========== old coordinates
# def build_bpca(num_pcles=128, radius=1., through_orig=False):
#     """Build a simple ballistic particle cluster aggregate by generating particle and
#      allowing it to stick where it first intersects another particle."""
#
#     sim = Simulation(max_pcles=num_pcles)
#     sim.add(Particle(x=0., y=0., z=0., r=radius), check=False)
#
#     # generate a "proposed" particle and trajectory, and see where it intersects the
#     # aggregate. add the new particle at this point!
#     for n in range(num_pcles-1):
#
#         success = False
#         while not success:
#
#             print 'Generating particle %d of %d' % (n+1, num_pcles)
#
#             if through_orig:
#                 second = np.array( [0.,0.,0.] )
#             else:
#                 second = random_sphere() * max(sim.farthest() * 2., radius *4.)
#
#             test_pt = random_sphere() * max(sim.farthest() * 2., radius *4.)
#             dist = np.linalg.norm(test_pt)
#             intersect = None
#             particle = None
#
#             for pcle in sim.particles:
#                 if debug: print 'Checking intersect with particle %d' % pcle.id
#                 i1, i2 = sphere_line((pcle.x, pcle.y, pcle.z), pcle.r, test_pt, second )
#                 if i1 is None and i2 is None:
#                     if debug: print('No intersection with particle %d' % pcle.id)
#                     continue
#                 else:
#                     i1 = 0 if None else i1
#                     i2 = 0 if None else i2
#                     if (np.linalg.norm(i1-test_pt) < np.linalg.norm(i2-test_pt) ):
#                         selected = i1
#                     else:
#                         selected = i2 # should always be the case here
#                     distance = np.linalg.norm(test_pt-selected)
#                     # look for the closest to the test point
#                     if distance < dist:
#                         if debug: print 'Intersect with pcle %d at distance %f' % (pcle.id, distance)
#                         dist = distance
#                         intersect = selected
#                         particle = pcle
#
#             if particle is None:
#                 success = False
#                 continue
#
#             # shift the origin along the line from the particle centre to the intersect
#             pcle_pos = np.array( [particle.x, particle.y, particle.z] )
#             shift = np.linalg.norm(intersect-pcle_pos)
#             vec = (intersect-pcle_pos) / shift
#             intersect += vec*shift
#
#             # Add to the simulation, checking for overlap with existing partilces (returns False if overlap detected)
#             success = sim.add(Particle(x=intersect[0], y=intersect[1], z=intersect[2], r=radius), check=True)
#             # if proposed particle is acceptable, add to the sim and carry on
#             if success & debug: print 'Adding particle at distance %f' % np.linalg.norm(intersect)
#
#     return sim
