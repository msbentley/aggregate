#!/usr/bin/python
"""
simulation.py

Mark S. Bentley (mark@lunartech.org), 2016

Simulation environment module."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist

debug = False

class Simulation:

    def __init__(self, max_pcles=1000):
        """Initialise the simulation. The key parameter here is the maximum number of particles,
        in order to pre-allocate array space.

        nattr=<int> can be used to set the number of additional attributes (besides
        x,y,z,r and id) to be stored in the array. Not all particle attributes
        need to be stored in the array, but those that may be queried for
        particle selection should be (for speed)."""

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
        return "<Simulation object contaning %d particles>" % ( self.count )

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


    def add(self, pos, radius, check=True):
        """
        Add a particle to the simulation. If check=True the distance between the proposed
        particle and each other is checked so see if they overlap. If so, False is returned.
        """

        if check:
            if not self.check(pcle):
                return False

        if len(pos) != 3:
            print('ERROR: particle position should be given as an x,y,z tuple')
            return None

        if len(radius) != 1:
            print('ERROR: particle radius must be a single value')
            return None

        self.pos[self.count] = pcle[0:3]
        self.radius[self.count] = pcle[3]
        self.mass[self.count] = (4./3.)*np.pi*pcle[3]

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


    def fractal_scaling(self, prefactor=1.27):

        # N = k * (Rg/a)**Df
        return np.log(self.count/prefactor)/np.log(self.gyration()/self.radius.min())


    def fractal_mass_radius(self, num_pts=100, show=False):
        """Calculate the fractal dimension of the domain using the relation:

        m(r) prop. r**D_m"""

        r = np.linalg.norm(self.pos, axis=1)
        start = self.radius.min()
        stop = (self.farthest()+self.radius.max())*.8
        step = (stop-start)/float(num_pts)
        radii = np.arange(start, stop, step)
        count = np.zeros(len(radii))

        # TODO - implement sphere-sphere intersection and correctly calculate
        # contribution from spheres at the boundary of the ROI.

        for idx, i in enumerate(radii):
            count[idx] =  r[r<=i].size

        # Need to fit up until the curve is influenced by the outer edge
        coeffs = np.polyfit(np.log(radii),np.log(count),1)
        poly = np.poly1d(coeffs)

        if show:
            fig, ax = plt.subplots()
            ax.loglog(radii, count)
            ax.grid(True)
            ax.set_xlabel('radius')
            ax.set_ylabel('count')
            yfit = lambda x: np.exp(poly(np.log(radii)))
            ax.loglog(radii, yfit(radii))
            plt.show()

        return coeffs[0]



    def fractal_box_count(self, num_grids=100):
        """Calculate the fractal dimension of the domain using the cube-counting method"""

        # need to determine if a square contains any of a sphere...
        # first use a bounding box method to filter the domain
        xmin = self.pos[:,0].min() - self.radius[np.argmin(self.pos[:,0])]
        xmax = self.pos[:,0].max() + self.radius[np.argmin(self.pos[:,0])]

        ymin = self.pos[:,1].min() - self.radius[np.argmin(self.pos[:,1])]
        ymax = self.pos[:,1].max() + self.radius[np.argmin(self.pos[:,1])]

        zmin = self.pos[:,2].min() - self.radius[np.argmin(self.pos[:,2])]
        zmax = self.pos[:,2].max() + self.radius[np.argmin(self.pos[:,2])]


    def to_csv(self, filename):
        np.savetxt(filename, self.pos, delimiter=",")
