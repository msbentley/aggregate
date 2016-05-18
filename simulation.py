#!/usr/bin/python
"""
simulation.py

Mark S. Bentley (mark@lunartech.org), 2016

Simulation environment module."""

import numpy as np
from scipy.spatial.distance import pdist, cdist

debug = False

# TODO: store particle data in numpy arrays. This should allow efficient selection,
# e.g. to do bounding box checking to see particles are potentially close to the new
# one when checking for overlaps.

class Simulation:

    def __init__(self, max_pcles=1000):
        """Initialise the simulation. The key parameter here is the maximum number of particles,
        in order to pre-allocate array space.

        nattr=<int> can be used to set the number of additional attributes (besides
        x,y,z,r and id) to be stored in the array. Not all particle attributes
        need to be stored in the array, but those that may be queried for
        particle selection should be (for speed)."""

        self.particles = []
        self.pos = np.zeros( (max_pcles, 3 ), dtype=np.float )
        self.id = np.zeros( max_pcles, dtype=np.int )
        self.radius =  np.zeros( max_pcles, dtype=np.float )
        self.mass = np.zeros( max_pcles, dtype=np.float )
        self.count = 0
        self.next_id = 0


    def __str__(self):
        """
        Returns a string with the number of particles and the bounding box size.
        """
        return "<Simulation object contaning %d particles>" % ( len(self.particles) )

    #__add__, __sub__, and __mul__


    def show(self, using='maya'):
        """A simple scatter-plot to represent the aggregate - either using mpl
        or mayavi"""

        if using=='mpl':

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], s=100.)
            plt.show()

        elif using=='maya':
            from mayavi.mlab import points3d
            points3d(self.pos[:,0], self.pos[:,1], self.pos[:,2], self.radius, scale_factor=2, resolution=16)

        return


    def add(self, pcle, check=True):
        """
        Add a particle to the simulation. If check=True the distance between the proposed
        particle and each other is checked so see if they overlap. If so, False is returned.
        """

        if check:
            if not self.check(pcle):
                return False

        self.particles.append(pcle)
        self.pos[self.count] = (pcle.x, pcle.y, pcle.z)
        self.radius[self.count] = pcle.r
        self.mass[self.count] = (4./3.)*np.pi*pcle.r**3.

        self.count += 1
        if pcle.id is None:
            pcle.id = self.next_id
            self.id[self.count-1] = pcle.id
            self.next_id += 1

        return True


    def intersect(self, position, direction, closest=True):

        # see, for example, https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

        # calculate the discriminator using numpy arrays
        vector = position - self.pos[0:self.count]
        b = np.sum(vector * direction, 1)
        c = np.sum(np.square(vector), 1) - self.radius[0:self.count] * self.radius[0:self.count]
        disc = b * b - c

        # disc<0 when no intersect, so ignore those cases
        possible = np.where(disc >= 0.0)
        # complete the calculation for the remaining points
        disc = disc[possible]
        ids = self.id[possible]

        if len(disc)==0:
            return None, None

        b = b[possible]
        # two solutions: -b/2 +/- sqrt(disc) - this solves for the distance along the line
        dist1 = -b - np.sqrt(disc)
        dist2 = -b + np.sqrt(disc)
        dist = np.minimum(dist1, dist2)

        # choose the minimum distance and calculate the absolute position of the hit
        hits = position + dist[:,np.newaxis] * direction

        if closest:
            return ids[np.argmin(dist)], hits[np.argmin(dist)]
        else:
            return ids, hits


    def check(self, pcle):

        pcle_arr = np.array( [(pcle.x, pcle.y, pcle.z)] )
        if (cdist(pcle_arr, self.pos[0:self.count+1])[0] < (pcle.r + self.radius[0:self.count+1].max())).sum() > 0:
            if debug: print('Cannot add particle here!')
            return False
        else:
            return True


    def farthest(self):
        return self.pos.max()


    def com(self):
        """Compute centre of mass"""

        return np.average(self.pos[:self.count], axis=0, weights=self.mass[:self.count])

    def gyration(self):
        """Returns the radius of gyration: the RMS of the monomer distances from the
        centre of mass of the aggregate"""

        return np.sqrt(np.square(self.pos[:self.count]-self.com()).sum()/self.count)

    def char_rad(self):
        """Calculates the characteristic radius:
        a_c = sqrt(5/3) * R_g"""

        return np.sqrt(5./3.) * self.gyration()

    def porosity(self):
        """Calculates porosity as 1 - vol / vol_gyration"""

        return (1. - ( (self.mass[:self.count].sum()) / ((4./3.)*np.pi*self.gyration()**3.) ) )


    def fractal_mass_radius(self, num_pts=100, prefactor=None, show=False):
        """Calculate the fractal dimension of the domain using the relation:

        m(r) prop. r**D_m"""





    def position(self):
        return np.array( [self.x, self.y, self.z] )


    def to_array(self, id=False):

        ndim = 4 if id else 3
        pcle_array = np.ndarray( (self.count,ndim) )
        for idx, pcle in enumerate(self.particles):
            pcle_array[idx,0:3] = (pcle.x, pcle.y, pcle.z)
            if id:
                pcle_array[idx,3] = pcle.id
        return pcle_array


    def to_csv(self, filename):
        np.savetxt(filename, self.pos, delimiter=",")


#===== old coordinates
    # def check_simple(self, pcle):
    #
    #     for p in self.particles:
    #         if debug: print('Comparing to pcle %d' %p.id)
    #         pvec = np.array( [p.x, p.y, p.z] )
    #         pclevec = np.array( [pcle.x, pcle.y, pcle.z] )
    #         if np.linalg.norm(pvec-pclevec) < (p.r+pcle.r):
    #             if debug: print '%3.2f, %3.2f' % (np.linalg.norm(pvec-pclevec), (p.r+pcle.r))
    #             if debug: print('Cannot add particle here!')
    #             return False
    #
    #     return True
