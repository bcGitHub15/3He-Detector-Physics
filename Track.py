#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:38:21 2026

@author: bcollett
"""
import numpy as np
import matplotlib.pyplot as plt
import random

Debug = False


# A direction is a pair of angles
class Direction:
    def __init__(self, theta:float, phi:float):
        self.theta = float(theta)
        self.phi = float(phi)


# We use points to represent either 3D points or 3D vectors
class Point3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if x > -100.0 and x < 100.0:
            self.x = x
        else:
            raise ValueError('x out of range +/-100 or not a number')
        if y > -100.0 and y < 100.0:
            self.y = y
        else:
            raise ValueError('y out of range +/-100 or not a number')
        if z > -100.0 and z < 100.0:
            self.z = z
        else:
            raise ValueError('z out of range +/-100 or not a number')
    
    def copy(self):
        return Point3D(self.x, self.y, self.z)
        

    def radius(self):
        return np.sqrt(self.x * self.x + self.y * self.y)
    
    def length(self):
        return np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
    
    def direction(self):
        r = self.length()
        nx = self.x / r
        ny = self.y / r
        nz = self.z / r
        return Point3D(nx, ny, nz)
    
    # Make printable
    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'
    
    # These modify the point and are likely more useful when using the
    # triple as a vector than as a point
    def divide_by(self, div:float):
        divisor = float(div)
        self.x /= divisor
        self.y /= divisor
        self.z /= divisor
        return self

    def times_by(self, mul:float):
        if Debug:
            print(self, "*", mul)
        multiplier = float(mul)
        self.x *= multiplier
        self.y *= multiplier
        self.z *= multiplier
        if Debug:
            print('-->', self)
        return self

    def plusab(self, p):
        if not isinstance(p, Point3D):
            raise ValueError('argument to plus must be of class point3D')
        if Debug:
            print(self, " + ", p)
        self.x += p.x
        self.y += p.y
        self.z += p.z
        if Debug:
            print('-->',self)
        return self        

    def minusab(self, p):
        if not isinstance(p, Point3D):
            raise ValueError('argument to minus must be of class point3D')
        self.x -= p.x
        self.y -= p.y
        self.z -= p.z
        return self

    # These generate new points from an old point and another value
    def divide(self, div:float):
        divisor = float(div)
        p = Point3D.from_point(self)
        p.x /= divisor
        p.y /= divisor
        p.z /= divisor
        return p

    def times(self, mul:float):
        if Debug:
            print(self, "*", mul)
        multiplier = float(mul)
        p = Point3D.from_point(self)
        p.x *= multiplier
        p.y *= multiplier
        p.z *= multiplier
        if Debug:
            print('-->', p)
        return p

    def plus(self, p):
        if not isinstance(p, Point3D):
            raise ValueError('argument to plus must be of class point3D')
        if Debug:
            print(self, " + ", p)
        np = Point3D.from_point(self)
        np.x += p.x
        np.y += p.y
        np.z += p.z
        if Debug:
            print('-->',np)
        return p        

    def minus(self, p):
        if not isinstance(p, Point3D):
            raise ValueError('argument to minus must be of class point3D')
        np = Point3D.from_point(self)
        np.x -= p.x
        np.y -= p.y
        np.z -= p.z
        return p
    
    # plot as a vector on an axis
    def plot_vec_at_on(self, p, ax, scale = 1.0):
        if not isinstance(p, Point3D):
            raise ValueError('First argument to plot_vec_at must be a point, found a', type(p))
        ratio = float(scale)
#        mag = float(length)
        xs = [p.x, p.x + self.x * ratio]
        ys = [p.y, p.y + self.y * ratio]
        zs = [p.z, p.z + self.z * ratio]
        ax.plot(xs, ys, zs, 'g-')

    # CLASS function to act as a generator like __init__ but from polar
    # parameters
    def from_spherical(r, theta, phi):
        p = Point3D()
        if r < -100.0 or r > 100.0:
            raise ValueError('radius out of range 0-100 or not a number')
        if theta >= -np.pi and theta <= np.pi:
            p.z = r * np.cos(theta)
        else:
            raise ValueError('theta out of range +/-pi or not a number', theta)
        if phi >= -2*np.pi and phi <= 2*np.pi:
            p.x = r * np.sin(theta) * np.cos(phi)
            p.y = r * np.sin(theta) * np.sin(phi)
        else:
            raise ValueError('phi out of range +/-2pi or not a number', phi)
        return p
    
    # and sim but passing in a direction instead of two angles
    def from_direction(r, d:Direction):
        if not isinstance(d, Direction):
            raise ValueError('In point_from_direction argument must be a Direction.')
        return Point3D.from_spherical(r, d.theta, d.phi)
    
    # essentially a copy constructor
    def from_point(p):
        if not isinstance(p, Point3D):
            raise ValueError('In from_point argument must be a Point3D.')
        return Point3D(p.x, p.y, p.z)
    


#
# A track is basically a pair of arrays, one containing positions and the
# the other charges. These represent the charge in each little chunks
# of the ionization track.
# Note that for debugging and clarity purposes I store the proton and triton
# tracks separately, though in practice they form a single trail of charge.
#
class Track:
    # Class parameters control such things as the number/length of segments
    # into which each sub-track is broken
    # 
    p_energy = 573_000 # eV
    p_range = 5.76e-3 # m
    t_energy = 191_000 # eV
    t_range = -1.92e-3 # m
    n_segment = 100
    E_pair = 45 # eV energy to generate a single electron-ion pair in He
    p_charge = p_energy/E_pair  # Total charge in proton track
    t_charge = t_energy/E_pair  # Total chare in triton track
    p_dx = p_range / n_segment
    t_dx = t_range / n_segment

    
    # Constructor
    def __init__(self, origin:Point3D, direction:Direction):
        if not isinstance(origin, Point3D):
            raise ValueError('First argument to Track must be a point, found a', type(origin))
        if not isinstance(direction, Direction):
            raise ValueError('second argument to Track must be a direction, found a', type(direction))
        # save copies of parameters mostly for debugging
        self.origin = origin
        self.dir = direction
        #
        # given direction, track length, and number of segments we can compute
        # the position increment for each track
        p_incr = Point3D.from_direction(Track.p_range, self.dir)
        p_incr.divide_by(Track.n_segment)
        t_incr = Point3D.from_direction(Track.t_range, self.dir)
        t_incr.divide_by(Track.n_segment)
#        print('origin', origin, 'pincr', p_incr, 'tincr', t_incr)
        #
        # Build arrays of points
        # Note that by adding i+0.5 * incr we make each point the center
        # of a little segement of the track
        self.p_positions = [None]*Track.n_segment
        self.p_charges = [None]*Track.n_segment
        self.t_positions = [None]*Track.n_segment
        self.t_charges = [None]*Track.n_segment
        p_tot = 0
        t_tot = 0
        for i in range(Track.n_segment):
            # Positions along the proton track
            pn = Point3D.from_point(self.origin)
            pn.plusab(p_incr.times(i+0.5))
#            print('<', n, '>', p_incr)
            self.p_positions[i] = pn
#            print(i, '--', self.p_positions[i])
            # positions along the triton track           
            tn = Point3D.from_point(self.origin)
            tn.plusab(t_incr.times(i+0.5))
            self.t_positions[i] = tn
#            print(i, '++', self.t_positions[i])
#        print('origin', origin, 'pincr', p_incr, 'tincr', t_incr)
        # charge densities along each track
        # f(xi) is the traction of charge density at xi
        xi = np.linspace(0, 1.0, num=Track.n_segment, endpoint=False) + 0.5/Track.n_segment
        df = 8.0 * np.power(1-xi, -5.0/13.0) / 13.0
        self.p_charges = Track.p_charge * df / Track.n_segment
        self.t_charges = Track.t_charge * df / Track.n_segment
            

    def print(self):
        for i in range(Track.n_segment):
            print(i, self.p_positions[i].x, self.p_positions[i].y, self.p_positions[i].z)

    # Plotters
    def plot_on(self, ax):
        pxs = np.zeros(Track.n_segment)
        pys = np.zeros(Track.n_segment)
        pzs = np.zeros(Track.n_segment)
        txs = np.zeros(Track.n_segment)
        tys = np.zeros(Track.n_segment)
        tzs = np.zeros(Track.n_segment)
        for i in range(Track.n_segment):
            pxs[i] = self.p_positions[i].x
            pys[i] = self.p_positions[i].y
            pzs[i] = self.p_positions[i].z
            txs[i] = self.t_positions[i].x
            tys[i] = self.t_positions[i].y
            tzs[i] = self.t_positions[i].z
#        print(xs, ys, zs)
        ax.scatter(pxs, pys, pzs, c=self.p_charges, cmap='Reds')
        ax.scatter(txs, tys, tzs, c=self.t_charges, cmap='Blues')

    def plot(self):
        fig1 = plt.figure()
        ax = fig1.add_subplot(projection='3d')
        self.plot_on(ax)
        return ax
    


# detector is a cylinder with its axis along the z direction and its base at z=0
# A neutron detector has to contain 3He but the pressure could be a parameter
# so we allow it as an optional argument
# Treat the wire diameter in the same way
# I choose to make the detector voltage an extrinsic property and set it
# with a method
# All of the arguments and properties are in SI units
class detector:
    def __init__(self, radius, length, pressure=1.01e6, rwire=2e-4):
        if radius > 0 and radius < 0.1:
            self.radius = radius
        else:
            raise ValueError('Radius out of range 0-0.1 or not a number')
        if length > 0 and length < 0.1:
            self.length = length
        else:
            raise ValueError('length out of range 0-0.1 or not a number')
        self.pressure = float(pressure)
        self.rwire = rwire
        # Build a bounding box to support decay creation
        self.zbound = 1.01 * self.length
        self.rbound = 1.01 * self.radius
        # pre-factors for field computations
        self.V0 = 0
        self.Er_pre = 0
        
    def set_voltage(self, voltage):
        self.V0 = float(voltage)
        self.V_pre = self.rwire * self.V0 / (self.radius - self.rwire)
        self.Er_pre = self.radius * self.rwire * self.V0 / (self.radius - self.rwire)

    # decide whether a point lies inside the detector
    def contains(self, p:Point3D):
        if not isinstance(p, Point3D):
            raise ValueError('argument to is_inside must be of class point3D')
        if p.radius() > self.radius:
            return False
        if p.z < 0 or p.z > self.length:
            return False
        return True

    # return a new point randomly located within the detector
    def new_point(self):
        while True:
            p = Point3D(random.uniform(-self.rbound, self.rbound),
                        random.uniform(-self.rbound, self.rbound),
                        random.uniform(0.0, self.zbound))
            if self.contains(p):
                return p

    # generate a random direction, uniformly distributed over 4pi
    def new_direction(self):
        while True:
            p = Point3D(random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, 1.0),
                        random.uniform(-1.0, 1.0))
            if p.radius() <= 1.0:
                break
        phi = np.arctan2(p.y, p.x)
        theta = np.arctan2(p.radius(), p.z)
        return Direction(theta, phi)
    
    # show a wireframe version
    def plot(self):
        fig1 = plt.figure()
        ax = fig1.add_subplot(projection='3d')
        npt = 10
        nseg = 8
        nring = 5
        nstep = 400
        xs = np.zeros(npt)
        ys = np.zeros(npt)
        # every outer line has the same z array
        zs = np.linspace(0, self.length, num=npt)
        for i in range(nseg):
            theta = 2 * np.pi * i / nseg
            xs.fill(self.radius * np.cos(theta))
            ys.fill(self.radius * np.sin(theta))
            ax.plot(xs, ys, zs, 'k-')
        # Add the central wire
        xs.fill(0)
        ys.fill(0)
        ax.plot(xs, ys, zs, 'k-')
        # Every ring has same x and y
        rtheta = np.linspace(0, 2 * np.pi, nstep, endpoint=False)
        rxs = self.radius * np.cos(rtheta)
        rys = self.radius * np.sin(rtheta)
        for i in range(nring+1):
            rzs = np.zeros(nstep) + i * self.length / nring
            ax.plot(rxs, rys, rzs, 'k-')
        return ax

    # return field at a given position, answer is in form of a Point3D
    def E_field_at(self, p:Point3D):
        if not isinstance(p, Point3D):
            raise ValueError('argument to E_field at must be of class point3D')
        r = p.radius()
        e_magr = self.Er_pre / (r ** 3)
        # direction is outward radially
        e = Point3D.from_point(p)
        if Debug:
            print('p=', p, 'r=',r,'E=',e)
        e.z = 0
        e.x *= e_magr
        e.y *= e_magr
        if Debug:
            print('r=',r,'E=',e)
        return e
        
    # return potential at a given position, answer is a float
    def V_at(self, p:Point3D):
        if not isinstance(p, Point3D):
            raise ValueError('argument to E_field at must be of class point3D')
        r = p.radius()
        return self.V_pre * (self.radius/r - 1)
    
    # find a drift velocity given an E field 


if __name__ == '__main__':
    det = detector(12.7e-3, 0.02)
    det.set_voltage(1400)
#    thePlot = det.plot()
    # Change if statements to enable tests
    # Test event position generation
    if False:
        xs=[]
        ys=[]
        zs=[]
        for i in range(1000):
            p = det.new_point()
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
        fig1 = plt.figure()
        ax = fig1.add_subplot(projection='3d')
        ax.scatter(xs, ys, zs)
    # test event direction generation
    if False:
        xd=[]
        yd=[]
        zd=[]
        for i in range(1000):
            nd = det.new_direction()
            p = Point3D.from_direction(1.0, nd)
    #        print(nd.theta, nd.phi, p.x, p.y, p.z)
            xd.append(p.x)
            yd.append(p.y)
            zd.append(p.z)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        ax2.scatter(xd, yd, zd)
    # test event generation
    if False:
        n_track = 20
        for i in range(n_track):
            p = det.new_point()
            nd = det.new_direction()
            t = Track(p, nd)
            t.plot_on(thePlot)
    # test efield computation and vector plotting
    if False:
        npt = 30
        thePlot = det.plot()
        for i in range(npt):
            while True:
                p = det.new_point()
                if p.radius() > 5e-4:
                    break
            e = det.E_field_at(p)
            print(e, p)
            e.plot_vec_at_on(p, thePlot, scale=1e-6)
    # Test track integration and gas amplification
    Debug1 = False
    Debug2 = False
    longest_step = 0.0
    nmax = 1000
    max_step = 2e-5
    min_step = 1.0
    start = Point3D(1.0e-3, -5.0e-3, 0.002)
    r = Point3D.from_point(start)
    print('init r', r)
    t = 0
    dt = 1.0e-7 # 100 ns
    n = 0
    track = [None]*nmax
    q = -1.6e-19
    gas_gain = 1e5
    Qmax = gas_gain * np.abs(q)
    Vpair = 45 # V
    Vstep = Vpair / 3 # Max voltage change for a single integration step
    print('Vpair=', Vpair)
    pair_fraction = 0.45
    n_charge_update= 0
    # Establish starting point
#    print('init r', r)
    track[0] = r
    init_radius = r.radius()
    #
    # We need to keep track of several voltages
    # V_new is the voltage after the most recent step
    # V_old is voltage last time the charge was updates
    # V_before is the voltage before we took the current step
    V_new = det.V_at(r)
    V_old = V_new
    print('Init V=', V_new)
    while True:
        efield = det.E_field_at(r)
        emag = efield.length()
        V_before = det.V_at(r)
        w = 35.767 * np.sqrt(emag)
        v = efield.direction().times_by(-w)
#        print('w, v, r', w, v, r)
        n_sub = 0
        while True:
            # Trial steps until we get one that is short enough
            '''
            # Step size control
            # make sure that w dt is under the max step
            # else reduce step size till it is.
            #
            speed = np.abs(w)
            step = dt * speed
            if step > longest_step:
                longest_step = step
            if step < min_step:
                min_step = step
            if step > max_step:
                dt = 0.5 * max_step / speed
                print('reduced step size', dt)
            '''
            # Trial step
            r_new = r.copy()
            r_new.x = r.x + v.x * dt
            r_new.y = r.y + v.y * dt
            r_new.z = r.z + v.z * dt
            new_radius = r_new.radius()
#            print('new position=',r_new, 'at radius', new_radius)
#            print('new radius', new_radius, dt, V_before)
            # test voltage change
            V_new = det.V_at(r_new)
            delta_V = np.abs(V_new - V_before)
#            print('>>>',V_new, V_before, delta_V, np.abs(V_new-V_old))
            if delta_V >= Vstep:
                # reduce step size and try again
                dt = dt / 3
                print('reduced step size', dt)
                n_sub += 1
            elif delta_V < 0.03*Vstep:
                dt = dt * 3
                print('increased step size', dt)
                n_sub+= 1
            else:
                # accept this step
#                print(f'Accept after {n_sub} cycles with dV={delta_V}')
                break
        # Converge test
        if new_radius <= 2.0e-4:
            break
        # Still going, update r
        r = r_new
        track[n] = Point3D.from_point(r)
#        new_r = r.radius()
        # see if update charge
        V_new = det.V_at(r)
        if np.abs(V_new - V_old) > Vpair:
            q  += pair_fraction * q
            n_charge_update += 1
            print('Update charge', V_new-V_old, '>', Vpair, np.abs(V_new - V_old) / Vpair)
            V_old = V_new
#        print(step, speed,V_new, V_old,'q',q)
        '''
        if Debug1:
            print('Old q=', q, 'old, new', V_old, V_new,)
#        old_dq = q * (V_old - V_new)/Epair
        dq = -q * emag * step * pair_fraction / Vpair
        print(step, speed,'q',q,'emag',emag,'factor',emag * step / Vpair)
        if np.abs(q) < Qmax:
            q += dq
        if Debug2:
            print(V_old, V_new, 'q=', q, 'dq=', dq)
        '''
        t += dt
        n = n+1
        if new_radius > init_radius:
            print(new_radius, '>', init_radius)
            print(track[:n])
            raise ValueError('Radius increased!')
        if n > nmax:
#        if n > 200:
            raise ValueError('Not converged after', n, 'steps')
    print('Converged in ', n, 'steps. Final dt=', dt)
    print('Longest step=', longest_step, 'shortest=', min_step)
    print(n_charge_update, 'charge updates')
    print(t, track[n-1], q/1.6e-19)
    pxs = np.zeros(nmax)
    pys = np.zeros(nmax)
    pzs = np.zeros(nmax)
    for i in range(n-1):
        pxs[i] = track[i].x
        pys[i] = track[i].y
        pzs[i] = track[i].z
#        print(xs, ys, zs)
    thePlot = det.plot()

    thePlot.scatter(pxs[:n], pys[:n], pzs[:n], '.')
    
   
    
    

    
