#!/usr/bin/python
"""
particle.py

Mark S. Bentley (mark@lunartech.org), 2016

Particle class.

"""

import numpy as np


class Particle:
    """particle class"""

    def __init__(self, x=0., y=0., z=0., r=1.):

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.r = float(r)
        self.id = None


    def __str__(self):
        """
        Returns a string with the position of the particle.
        """
        return "<Particle object, x=%s y=%s z=%s>"%(self.x,self.y,self.z)


    def distance(self, orig=(0.,0.,0.)):

        return np.sqrt(
            (self.x-float(orig[0]))**2. +
            (self.y-float(orig[1]))**2. +
            (self.z-float(orig[2]))**2. )


    def move(self, dx, dy, dz):

        self.x += float(dx)
        self.y += float(dy)
        self.z += float(dz)


    def move_along(self, p1, p2, dist):
        """
        Moves the particle along the line given by p1 and p2 by distance dist
        """
        vector = p2-p1
        vector /= np.sqrt( vector**2. )
        position = dist * vector
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
