#!/usr/bin/python
"""
impact.py

Mark S. Bentley (mark@lunartech.org), 2016

A set of YADE scripts to simulate the impact of an aggregate particle
onto a target, given input physical parameters.

"""

from yade import pack, plot

# Create an initial particle cloud
sp = pack.SpherePack()
numpart = sp.makeCloud( (-0.01,-0.01,0.0001), (0.01,0.01,0.0201), rMean=0.0005) # = 1401 pcls
print str(numpart) + ' particles created'
sp.toSimulation()

# Add a floor (plane in the Y axis at zero height)
O.bodies.append(utils.wall(0,axis=2))

# Add some physics, with default material properties and gravity!
O.engines=[
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb(),Bo1_Wall_Aabb()]),
    InteractionLoop(
        # Account for sphere-sphere and sphere-wall collisions
        [Ig2_Sphere_Sphere_L3Geom(),Ig2_Wall_Sphere_L3Geom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_L3Geom_FrictPhys_ElPerfPl()] ),
    GravityEngine(gravity=(0,0,-9.81),label='gravity'),
    NewtonIntegrator(damping = 0.2,label='newton'),
    PyRunner(command='record()',realPeriod=0.01,label='runner'), # call "record" every 0.01 s of sim time
    PyRunner(command='checkUnbalanced()',realPeriod=5)
]

# Set initial timestep to half the critical timestep
O.dt=0.5*utils.PWaveTimeStep()

print 'Initial timestep = ' + str(O.dt)

# Plot total kinetic energy versus time
plot.plots={'t':('coordNum',None,'Ek')}

def record():
    # Use the built-in "addData" function to store basic data at every timestep
    plot.addData(i=O.iter,t=O.time,Ek=utils.kineticEnergy(),coordNum=utils.avgNumInteractions())

# Stop the sim once we've reached an ~static state (no more unbalanced forces)
def checkUnbalanced():
    print 'Unbalanced force: ' + str(utils.unbalancedForce())
    if utils.unbalancedForce() < 0.05:
        O.pause()

# Enable energy tracking
O.trackEnergy=True

# show the plot on the screen, and update while the simulation runs
plot.plot(subPlots=False)
# O.run()

# Saving temporary state for messing around
O.saveTmp()
