#!/usr/bin/python
"""
simulation.py

Mark S. Bentley (mark@lunartech.org), 2016

Simulation environment module."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class Simulation:
    """
    This class provides functionality for a generic particle simulation, with methods for
    adding particles, moving them in the simulation domain, checking for intersects and
    overlaps etc. It also calculates basic properties of particle collections within the domain.
    """

    def __init__(self, max_pcles=1000, filename=None, density=None, debug=False):
        """
        Initialise the simulation. The key parameter here is the maximum number of particles,
        in order to pre-allocate array space.

        nattr=<int> can be used to set the number of additional attributes (besides
        x,y,z,r and id) to be stored in the array. Not all particle attributes
        need to be stored in the array, but those that may be queried for
        particle selection should be (for speed).
        """

        if (max_pcles is None) and (filename is None):
            print('WARNING: simulation object created, but no size given - use max_pcles or sim.from_csv')
        elif filename is not None:
            self.from_csv(filename, density=density)
        else:
            self.pos = np.zeros( (max_pcles, 3 ), dtype=np.float )
            self.id = np.zeros( max_pcles, dtype=np.int )
            self.radius =  np.zeros( max_pcles, dtype=np.float )
            self.volume = np.zeros( max_pcles, dtype=np.float )
            self.mass = np.zeros( max_pcles, dtype=np.float )
            self.density = np.zeros( max_pcles, dtype=np.float )
            self.count = 0
            self.agg_count = 0
            self.next_id = 0

        self.debug = debug


    def __str__(self):
        """
        Returns a string with the number of particles and the bounding box size.
        """
        return "<Simulation object contaning %d particles>" % ( self.count )



    def get_bb(self):
        """
        Return the bounding box of the simulation domain
        """

        xmin = self.pos[:,0].min() - self.radius[np.argmin(self.pos[:,0])]
        xmax = self.pos[:,0].max() + self.radius[np.argmin(self.pos[:,0])]

        ymin = self.pos[:,1].min() - self.radius[np.argmin(self.pos[:,1])]
        ymax = self.pos[:,1].max() + self.radius[np.argmin(self.pos[:,1])]

        zmin = self.pos[:,2].min() - self.radius[np.argmin(self.pos[:,2])]
        zmax = self.pos[:,2].max() + self.radius[np.argmin(self.pos[:,2])]

        return (xmin, xmax), (ymin,ymax), (zmin, zmax)


    def bb_aspect(self):
        """
        Returns the aspect ratio X:Y:Z of the bounding box.
        """

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()
        xsize = xmax-xmin
        ysize = ymax-ymin
        zsize = zmax-zmin

        return (xsize, ysize, zsize)/min(xsize, ysize, zsize)


    def fit_ellipse(self, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points

        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!

        Code adapted from: https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py

        Returns:
        (center, radii, rotation)

        """

        from numpy import linalg

        P = self.pos

        (N, d) = np.shape(P)
        d = float(d)

        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)])
        QT = Q.T

        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse
        center = np.dot(P.T, u)

        # the A matrix for the ellipse
        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) -
                       np.array([[a * b for b in center] for a in center])
                       ) / d

        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)

        return (center, radii, rotation)




    def elongation(self):

        (center, radii, rotation) = self.fit_ellipse()
        return max(radii/min(radii))


    def chull(self):
        """
        Calculates the convex hull (minimum volume) bounding the set of
        sphere centres - DOES NOT ACCOUNT FOR RADII!
        """

        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.pos)
        return hull



    def show(self, using='maya', fit_ellipse=False, show_hull=False):
        """
        A simple scatter-plot to represent the aggregate - either using mpl
        or mayavi
        """

        if fit_ellipse:

            (center, radii, rotation) = self.fit_ellipse()

            u = np.linspace(0.0, 2.0 * np.pi, 100)
            v = np.linspace(0.0, np.pi, 100)

            # cartesian coordinates that correspond to the spherical angles:
            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
            # rotate accordingly
            for i in range(len(x)):
                for j in range(len(x)):
                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

        if show_hull:
            hull = self.chull()
            hull_x = hull.points[:,0]
            hull_y = hull.points[:,1]
            hull_z = hull.points[:,2]

        if using=='mpl':

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_aspect('equal')
            ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2], s=100.)

            if fit_ellipse:
                ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='k', alpha=0.2)
            plt.show()

        elif using=='maya':
            import mayavi.mlab as mlab
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            mlab.points3d(self.pos[:,0], self.pos[:,1], self.pos[:,2], self.radius, scale_factor=2, resolution=16)

            if fit_ellipse:
                mlab.mesh(x,y,z, opacity=0.25, color=(1,1,1))

            if show_hull:
                mlab.triangular_mesh(hull_x, hull_y, hull_z, hull.simplices, representation='wireframe',color=(1,1,1))

        return



    def add(self, pos, radius, density=1., check=False):
        """
        Add a particle to the simulation.

        If check=True the distance between the proposed particle and each other
        is checked so see if they overlap. If so, False is returned.
        """

        if check:
            if not self.check(pos, radius):
                return False

        if len(pos) != 3:
            print('ERROR: particle position should be given as an x,y,z tuple')
            return None

        radius = float(radius)

        self.pos[self.count] = np.array(pos)
        self.radius[self.count] = radius
        self.volume[self.count] = (4./3.)*np.pi*radius**3.
        self.density[self.count] = density
        self.mass[self.count] = self.volume[self.count] * density

        self.count += 1
        self.id[self.count-1] = self.next_id
        self.next_id += 1

        return True



    def add_agg(self, sim, check=False):
        """
        Add an aggregate particle to the simulation. If check=True the distance between the proposed
        particle and each other is checked so see if they overlap. If so, False is returned.
        """

        if check:

            pass # TODO

            if not self.check(pos, radius):
                return False

        # TODO

        num_pcles = sim.count

        self.pos[self.count:self.count+num_pcles] = sim.pos[0:sim.count]
        self.radius[self.count:self.count+num_pcles] = sim.radius[0:sim.count]
        self.density[self.count:self.count+num_pcles] = sim.density[0:sim.count]
        self.volume[self.count:self.count+num_pcles] = (4./3.)*np.pi*sim.radius[0:sim.count]**3.
        self.mass[self.count:self.count+num_pcles] = \
            self.volume[self.count:self.count+num_pcles] * self.density[self.count:self.count+num_pcles]

        self.id[self.count:self.count+num_pcles] = self.id.max()+1+range(num_pcles)
        self.count += num_pcles

        return True



    def intersect(self, position, direction, closest=True):
        """
        Wrapper for line_sphere() that detects if the position passed is for a
        monomer or an aggregates and handles each case.
        """

        if type(position)==tuple:
            position = np.array(position)

        if len(position.shape)==2: # position is an array, i.e. an aggregate
            # loop over each monomer in the passed aggregate and check if it
            # intersects any of the monomers already in the domain

            # TODO: find a better, vectorised way to do this!'

            max_dist = 10000. # TODO calculate a sensible value here
            sim_id = None
            hits = None
            monomer_pos = None

            for pos in position:

                ids, dist = self.line_sphere(pos, direction, closest=True, ret_dist=True)
                if dist is not None:
                    if dist < max_dist:
                        max_dist = dist
                        monomer_pos = pos
                        sim_id = ids # id of the simulation agg

            if sim_id is not None:
                hit = monomer_pos + max_dist * direction # position of closest intersect
                return sim_id, max_dist, hit
            else:
                return None, None, None

        else:

            ids, hits = self.line_sphere(position, direction, closest=closest, ret_dist=False)

        return ids, hits


    def line_sphere(self, position, direction, closest=True, ret_dist=False):
        """
        Accepts a position and direction vector defining a line and determines which
        particles in the simulation intersect this line, and the locations of these
        intersections. If closest=True only the shortest (closest) intersect is
        returned, otherwise all values are given.

        If ret_dist=True then the distance from position to the hit will be returned,
        rather than the coordinates of the hit itself.

        See, for example, https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        """

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
            if ret_dist:
                return ids[np.argmin(dist)], dist[np.argmin(dist)]
            else:
                return ids[np.argmin(dist)], hits[np.argmin(dist)]
        else:
            if ret_dist:
                return ids, dist
            else:
                return ids, hits



    def check(self, pos, radius):
        """
        Accepts a proposed particle position and radius and checks if this overlaps with any
        particle currently in the domain. Returns True if the position is acceptable or
        False if not.
        """

        if len(pos.shape)==2: # passed an aggregate

            if cdist(pos, self.pos[0:self.count]).min() < (radius.max() + self.radius[0:self.count].max()) > 0:
                # TODO does not properly deal with polydisperse systems
                if self.debug: print('Cannot add aggregate here!')
                return False
            else:
                return True

        else:

            if (cdist(np.array([pos]), self.pos[0:self.count+1])[0] < (radius + self.radius[0:self.count+1].max())).sum() > 0:
                if self.debug: print('Cannot add particle here!')
                return False
            else:
                return True



    def farthest(self):
        """
        Returns the centre of the particle farthest from the origin
        """

        return self.pos.max()



    def com(self):
        """
        Computes the centre of mass
        """

        return np.average(self.pos[:self.count], axis=0, weights=self.mass[:self.count])



    def recentre(self):
        """
        Re-centres the simulation such that the centre-of-mass of the assembly
        is located at the origin (0,0,0)
        """

        self.pos -= self.com()



    def move(self, vector):
        """Move all particles in the simulation by the given vector"""

        self.pos += vector



    def gyration(self):
        """
        Returns the radius of gyration: the RMS of the monomer distances from the
        centre of mass of the aggregate.
        """

        return np.sqrt(np.square(self.pos[:self.count]-self.com()).sum()/self.count)



    def char_rad(self):
        """
        Calculates the characteristic radius: a_c = sqrt(5/3) * R_g
        """

        return np.sqrt(5./3.) * self.gyration()



    def porosity_gyro(self):
        """
        Calculates porosity as 1 - vol / vol_gyration
        """

        return (1. - ( (self.volume[:self.count].sum()) / ((4./3.)*np.pi*self.gyration()**3.) ) )


    def porosity_chull(self):

        return (1. - ( (self.volume[:self.count].sum()) / self.chull().volume ) )



    def density_gyro(self):
        """
        Calculates density as (mass of monomers)/(volume of gyration)
        """

        return (self.mass[:self.count].sum() / ((4./3.)*np.pi*self.gyration()**3.) )


    def density_chull(self):

        return (self.mass[:self.count].sum() / self.chull().volume)


    def fractal_scaling(self, prefactor=1.27):
        """Calculates the fractal dimension according to the scaling relation:
        N = k * (Rg/a)**Df
        The value of k, the fratcal pre-factor, can be set with prefactor=
        """

        return np.log(self.count/prefactor)/np.log(self.gyration()/self.radius[0:self.count].min())



    def fractal_mass_radius(self, num_pts=100, show=False):
        """
        Calculate the fractal dimension of the domain using the relation:

        m(r) prop. r**D_m
        """

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
        """
        Calculate the fractal dimension of the domain using the cube-counting method.
        """

        # need to determine if a square contains any of a sphere...
        # first use a bounding box method to filter the domain
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()

        # need to determine minimum and maximum box sizes, and the increments
        # to use when slicing the domain

        r_min = self.radius.min()
        max_div_x = int(np.ceil((xmax-xmin) / r_min))
        max_div_y = int(np.ceil((ymax-ymin) / r_min))
        max_div_z = int(np.ceil((zmax-zmin) / r_min))

        # calculate:
        # Df = Log(# cubes covering object) / log( 1/ box size)

        # pseudo-code from http://paulbourke.net/fractals/cubecount/
       # for all offsets
       #    for all box sizes s
       #       N(s) = 0
       #       for all box positions
       #          for all voxels inside the current box
       #             if the voxel is part of the object
       #                N(s) = N(s) + 1
       #                stop searching voxels in the current box
       #             end
       #          end
       #       end
       #    end
       # end



    def to_csv(self, filename):
        """
        Write key simulation variables (id, position and radius) to a CSV file
        """

        headertxt = 'id, x, y, z, radius'
        np.savetxt(filename, np.hstack( (self.id[:,np.newaxis], self.pos, self.radius[:,np.newaxis]) ),
            delimiter=",", header=headertxt)

        return



    def from_csv(self, filename, density=None):
        """
        Initialise simulation based on a file containing particle ID, position and radius.

        Note that particles with the same ID will be treated as members of an aggregate.
        """

        simdata = np.genfromtxt(filename, comments='#', delimiter=',')
        self.id = simdata[:,0].astype(np.int)
        self.pos = simdata[:,1:4]
        self.radius = simdata[:,4]
        if density is None:
            self.density = simdata[:,5]
        else:
            self.density = np.ones_like(self.id, dtype=float)
        self.volume = (4./3.)*np.pi*self.radius**3.
        self.mass = self.volume * self.density
        self.count = len(self.id)
        self.next_id = self.id.max()+1
        self.agg_count = len(np.unique(self.id))

        return



    def to_vtk(self, filename):
        """
        Writes the simulation domain to a VTK file. Note that evtk is required!
        """

        from evtk.hl import pointsToVTK

        # x = np.ascontiguousarray(self.pos[:,0])
        # y = np.ascontiguousarray(self.pos[:,1])
        # z = np.ascontiguousarray(self.pos[:,2])

        x = np.asfortranarray(self.pos[:,0])
        y = np.asfortranarray(self.pos[:,1])
        z = np.asfortranarray(self.pos[:,2])
        radius = np.asfortranarray(self.radius)
        mass = np.asfortranarray(self.mass)
        id = np.asfortranarray(self.id)

        pointsToVTK(filename, x, y, z,
            data = {"id" : id, "radius" : radius, "mass": mass})

        return



    def to_liggghts(self, filename, density=1000.):
        """
        Write to a LIGGGHTS data file, suitable to be read into a simulation.
        """

        # Save to a LAMMPS/LIGGGHTS data file, compatible with the read_data function
        #
        # Output format needs to be:
        # 42 atoms
        #
        # 1 atom types
        #
        # -0.155000000000000 0.155000000000000 xlo xhi
        # -0.155000000000000 0.155000000000000 ylo yhi
        # -0.150000000000000 1.200000000000000 zlo zhi
        #
        # Atoms
        #
        # 1 1 0.01952820 0.14099100 1.10066000 0.01073252 1000.0 1
        # 2 1 0.01811800 0.14345470 1.10433955 0.00536626 1000.0 1
        #
        # atom-ID atom-type x y z diameter density molecule-ID
        #
        # etc.

        liggghts_file = open('data.' + filename, 'w')
        liggghts_file.write('# LAMMPS data file\n\n')

        # TODO: update for aggregates once that code is in place

        liggghts_file.write(str(self.count) + ' atoms \n\n')
        liggghts_file.write('1 atom types\n\n')

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()

        liggghts_file.write(str(xmin) + ' ' + str(xmax) + ' xlo xhi\n')
        liggghts_file.write(str(ymin) + ' ' + str(ymax) + ' ylo yhi\n')
        liggghts_file.write(str(zmin) + ' ' + str(zmax) + ' zlo zhi\n\n')

        liggghts_file.write('Atoms\n\n')

        for idx in range(self.count):

            liggghts_file.write(str(self.id[idx]) + ' ' + str(1) + ' ' + str(2.*self.radius[idx]) + ' ' +
                str(density) + ' ' + str(self.pos[idx,0]) + ' ' + str(self.pos[idx,1]) + ' ' + str(self.pos[idx,2]) + '\n')

        liggghts_file.close()

        return


    def projection(self, xpix=512, ypix=512, vector=(0.,0.,-1.), show=False, png=None):
        """
        Produces a binary projection of the simulation with the number of
        pixels specified by xpix and ypix along the direction given by
        vector.

        If png= is set to a filename, a 2D graphic will be output.
        If show=True the image will be displayed.
        """

        binary_image = np.zeros( (xpix,ypix), dtype=bool )
        farthest = self.farthest() + 2.*self.radius.max()

        xs = np.linspace(-farthest, farthest, xpix)
        ys = np.linspace(-farthest, farthest, ypix)

        # TODO select out points that are in proxity to pixel projection

        # TODO: rotate points (about origin) such that the normal of the plane
        # matches vector

        for y_idx in range(ypix):
            for x_idx in range(xpix):
                pcle_id, intersect = self.intersect( (xs[x_idx], ys[y_idx], farthest), vector, closest=True )
                if intersect is not None:
                    binary_image[y_idx,x_idx] = True

        if png is not None or show:

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.imshow(binary_image, cmap=plt.cm.binary, extent=[-farthest, farthest, -farthest, farthest])

            if png is not None:
                fig.savefig(png)

            if show:
                plt.show()

        return binary_image


    def to_afm(self, xpix=256, ypix=256):
        """
        Assuming an infinitely sharp tip, this routine samples the simulation domain
        (up to the bounding box) at regular points in the XY plane, defined by the
        number of x and y pixels given by xpix and ypix.

        A 2D height field is returned which gives a simulated AFM image at the given
        resolution. Note that the 'substrate' is assumed simply to be the lowest
        point in the aggregate and values with no particle will be set to zero there."""

        afm_image = np.zeros( (xpix,ypix), dtype=np.float )

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()

        height = zmax + self.radius.max()

        xs = np.linspace(xmin, xmax, xpix)
        ys = np.linspace(ymin, ymax, ypix)

        for y_idx in range(ypix):
            for x_idx in range(xpix):
                pcle_id, intersect = self.intersect( (xs[x_idx], ys[y_idx], height), (0.,0.,-1.), closest=True )
                if intersect is None:
                    afm_image[y_idx,x_idx] = zmin
                else:
                    afm_image[y_idx,x_idx] = intersect[2]

        afm_image -= afm_image.min()

        return afm_image


    def to_gsf(self, filename, xpix=256, ypix=256):
        """
        Generates an AFM height field using to_afm() and writes to a Gwyddion simple
        field file (.gsf). The full file description can be found at:

        http://gwyddion.net/documentation/user-guide-en/gsf.html

        """

        import os, struct

        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self.get_bb()
        afm_image = self.to_afm(xpix=xpix, ypix=ypix)

        if os.path.splitext(filename) != '.gsf':
            filename += '.gsf'

        gsf_file = open(filename, 'w')
        gsf_file.write('Gwyddion Simple Field 1.0\n')
        gsf_file.write('XRes = %d\n' % xpix)
        gsf_file.write('YRes = %d\n' % ypix)
        gsf_file.write('XReal = %5.3f\n' % (xmax-xmin))
        gsf_file.write('YReal = %5.3f\n' % (ymax-ymin))

        gsf_file.close()

        # pad to the nearest 4-byte boundary with NULLs
        filesize = os.path.getsize(filename)
        padding = filesize % 4
        pad = struct.Struct('%dx' % padding)

        gsf_file = open(filename, 'ab')
        gsf_file.write(pad.pack())

        # Data values are stored as IEEE 32bit single-precision floating point numbers,
        # in little-endian (LSB, or Intel) byte order. Values are stored by row, from top to bottom,
        # and in each row from left to right.
        s = struct.pack('<%df' % (xpix*ypix), *np.ravel(afm_image).tolist())
        gsf_file.write(s)
        gsf_file.close()

        return



    def rotate(self, vector, axis=0):
        """
        Rotates the entire simulation (typically an aggregate centred on
        the origin) about the origin. Usually this is used to provide a
        random orientation. Inputs are a unit vector and an axis (0-3=X/Y/Z).

        A rotation matrix will be calculated between the specified axis and the
        given vector, and this will be applied to the particles in the simulation.
        """




def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.

    Taken from: http://svn.gna.org/svn/relax/tags/1.3.4/maths_fns/rotation_matrix.py

    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.

    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::

              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |


    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)
