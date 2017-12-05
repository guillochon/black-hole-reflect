from pylab import *
import numpy as numpy

# unit conversion: (use astropy for this?)
radians = numpy.pi/180.   # deg to radians
meters = 2.59*10**13.   # light days to meters
kg = 1.999*10.**30.     # kg/solar mass
grav = 6.67408*10**(-11.)	# m^3/kg/s^2 gravitational constant

def star_position(time):
		# returns the star positions for a given time
		# load in the star position file here:
		
		# calculate the position in cartesian coords:
		
		# random positions in a 20x20x20 box = x, y, z
		positions = transpose(array([rand(10)*20.-10., rand(10)*20.-10., rand(10)*20.-10.]))
		#print positions
		#print positions[:,0]  # all the x-coords
		#print positions[0,:]  # x,y,z of the first star  
		return positions   

def star_luminosity():
		# returns the star luminosities
		# load in the star luminosity file here:
		
		luminosities = rand(10)   # random numbers for testing
		#print luminosities
		return luminosities
		
def rotate(x, y, c, s):
		xx = c*x + s*y
		yy = -s*x + c*y
		x = xx
		y = yy
		return [x, y]
		
def gas_model(num_clouds, params, other_params):
		[mu, F, beta, theta_i, theta_o, kappa, mbh, f_ellip, f_flow, theta_e] = params
		[angular_sd_orbiting, radial_sd_orbiting, angular_sd_flowing, radial_sd_flowing] = other_params
		
		# First calculate the geometry of the emission:
		r = mu*F + (1.-F)*mu*beta**2.*numpy.random.gamma(beta**(-2.), 1, num_clouds)
		phi = 2.*numpy.pi*rand(num_clouds)
		x = r*cos(phi)
		y = r*sin(phi)
		z = r*0.
		
		
		angle = arccos(cos(theta_o) + (1. - cos(theta_o))*rand(num_clouds))  #*pow(u3[i], openingBendPower));
		cos1 = cos(angle)	   
		sin1 = sin(angle)
		u1 = rand(num_clouds)
		cos2 = cos(2.*numpy.pi*u1) 
		sin2 = sin(2.*numpy.pi*u1)
		cos3 = cos(0.5*numpy.pi - theta_i);
		sin3 = sin(0.5*numpy.pi - theta_i);
		
		# rotate to puff up:
		[x, z] = rotate(x, z, cos1, sin1)
		# rotate to restore axisymmetry:
		[x, y] = rotate(x, y, cos2, sin2)
		# rotate to apply inclination angle:
		[x, z] = rotate(x, z, cos3, sin3)
		
		w = 0.5 + kappa*x/sqrt(x*x + y*y + z*z)   #weights for the different points
		w /= sum(w)
		
		ptsize = w*15*num_clouds  # larger points correspond to more emission from the point
		shade = 0.5
		clf()
		subplot(2,2,1)  # edge-on view 1, observer at +infinity of x-axis
		scatter(x/meters, y/meters, ptsize, alpha=shade)
		xlabel("x")
		ylabel("y")
		subplot(2,2,2)  # edge-on view 2, observer at +infinity of x-axis
		scatter(x/meters, z/meters, ptsize, alpha=shade)
		xlabel("x")
		ylabel("z")
		subplot(2,2,3)   # view of observer looking at plane of sky
		scatter(y/meters, z/meters, ptsize, alpha=shade)
		xlabel("y")
		ylabel("z")
		subplot(2,2,4)   # plot the radial distribution of emission
		hist(r/meters,100)
		xlabel("r")
		ylabel("p(r)")
		show()
		
		# Now calculate velocities of the emitting gas:
		radius1 = sqrt(2.*grav*mbh/r)
		radius2 = sqrt(grav*mbh/r)
		vr = copy(x)*0.
		vphi = copy(x)*0.
		
		u5 = rand(num_clouds)
		n1 = numpy.random.normal(size=num_clouds)
		n2 = numpy.random.normal(size=num_clouds)
		for i in xrange(0, num_clouds):
				if u5[i] < f_ellip:
						# we give this point particle a near-circular orbit
						theta = 0.5*numpy.pi + angular_sd_orbiting*n1[i]
						vr[i] = radius1[i]*cos(theta)*exp(radial_sd_orbiting*n2[i])
						vphi[i] = radius2[i]*sin(theta)*exp(radial_sd_orbiting*n2[i])
				else:
						if f_flow < 0.5: 
								# we give this point particle an inflowing orbit
								theta = numpy.pi - theta_e + angular_sd_flowing*n1[i]
								vr[i] = radius1[i]*cos(theta)*exp(radial_sd_flowing*n2[i])
								vphi[i] = radius2[i]*sin(theta)*exp(radial_sd_flowing*n2[i])
						else:
								# we give this point particle an outflowing orbit
								theta = 0. + theta_e + angular_sd_flowing*n1[i]
								vr[i] = radius1[i]*cos(theta)*exp(radial_sd_flowing*n2[i])
								vphi[i] = radius2[i]*sin(theta)*exp(radial_sd_flowing*n2[i])

		# Convert vr, vphi to Cartesians:
		vx = vr*cos(phi) - vphi*sin(phi)
		vy = vr*sin(phi) + vphi*cos(phi)
		vz = vr*0.

		# rotate to puff up:
		[vx, vz] = rotate(vx, vz, cos1, sin1)
		# rotate to restore axisymmetry:
		[vx, vy] = rotate(vx, vy, cos2, sin2)
		# rotate to apply inclination angle:
		[vx, vz] = rotate(vx, vz, cos3, sin3)
		
		return [x, y, z, w, vx, vy, vz]
		



# Set constants:
num_clouds = 10000

mu = 5.   # mean radius of emission, in units of light days
F = 0.5   # minimum radius of emission, in units of fraction of mu 
beta = 0.5   # Gamma distribution radial profile shape parameter, between 0.01 and 2
theta_i = 70.   # inclination angle in deg, 0 deg is face-on
theta_o = 5.	# opening angle of disk in deg, 0 deg is thin disk, 90 deg is sphere
kappa = -0.5	# -0.5 emit back to center, 0 = isotropic emission, 0.5 emit away from center
log_mbh = 7.    # log10(black hole mass) in solar masses
f_ellip = 0.1	# fraction of particles in near-circular orbits
f_flow = 0.2	# 0-0.5 = inflow, 0.5-1 = outflow of fraction (1-f_ellip) of particles
theta_e = 20.	# angle in plane of v_r and v_phi, 0 -> radial inflow/outflow, 90 -> near-circular orbits

angular_sd_orbiting = 0.01
radial_sd_orbiting = 0.01
angular_sd_flowing = 0.01
radial_sd_flowing = 0.01

params = [mu*meters, F, beta, theta_i*radians, theta_o*radians, kappa,
				kg*10**log_mbh, f_ellip, f_flow, theta_e]  # work in SI units
other_params = [angular_sd_orbiting, radial_sd_orbiting, angular_sd_flowing, radial_sd_flowing]



# Calculate things:
star_positions = star_position(0)
star_luminosities = star_luminosity()
gas_coords = gas_model(num_clouds,params, other_params)




